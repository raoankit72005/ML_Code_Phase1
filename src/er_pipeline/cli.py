"""Connected data preparation, LightGBM training, scoring and submission CLI."""
import argparse
import hashlib
import json
import time
from pathlib import Path

from . import indexing, blocking, labeling, features, tables, evaluation
from .common import DEFAULTS, dump_json, find_source, find_truth

STAGES = ['index', 'block', 'label', 'features', 'export']
OUTPUTS = {
    'index': ['index.sqlite', 'index_report.json'],
    'block': ['candidate_pairs.parquet', 'queries.tsv.gz', 'blocking_report.json'],
    'label': ['labeled_pairs.parquet', 'query_labels.tsv.gz', 'candidate_recall.json', 'truth.sqlite'],
    'features': ['pair_features.parquet', 'feature_columns.json'],
    'export': ['candidate_pairs.tsv', 'tables_report.json'],
}


def fingerprint(path):
    path = Path(path).resolve()
    return dict(path=str(path), bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)


def prepare(args):
    config = dict(DEFAULTS)
    if args.config:
        supplied = json.loads(args.config.read_text(encoding='utf-8'))
        unknown = supplied.keys() - config.keys()
        if unknown:
            raise ValueError('Unknown config settings: ' + str(unknown))
        config.update(supplied)
    if not 0 < config['validation_fraction'] < 1:
        raise ValueError('validation_fraction must be strictly between 0 and 1')
    for k in config:
        if k == 'validation_fraction':
            continue
        if not isinstance(config[k], int) or (k != 'seed' and config[k] <= 0):
            raise ValueError(f'{k} must be an integer' + ('' if k == 'seed' else ' > 0'))
    if not 12 <= config['hash_bits'] <= 22:
        raise ValueError('hash_bits must be between 12 and 22')
    if config['lexical_top_k'] > config['lexical_pool']:
        raise ValueError('lexical_top_k must not exceed lexical_pool')
    if args.max_queries is not None and args.max_queries <= 0:
        raise ValueError('--max-queries must be positive')
    root = args.work.resolve()
    work = root / args.split
    work.mkdir(parents=True, exist_ok=True)
    tfidf = root / 'train' / 'tfidf.npz'
    if args.split == 'test' and not tfidf.exists():
        raise ValueError('Prepare the train index first; test must reuse its frozen TF-IDF statistics')
    inputs = [fingerprint(find_source(args.cleaned, args.split, n)) for n in (1,2,3)]
    truth_path = find_truth(args.cleaned) if args.split == 'train' else None
    if truth_path:
        inputs.append(fingerprint(truth_path))
    if args.split == 'test':
        inputs.append(fingerprint(tfidf))
    digest = hashlib.sha256()
    for p in sorted(Path(__file__).parent.glob('*.py')):
        digest.update(p.read_bytes())
    signature = dict(config=config, split=args.split, max_queries=args.max_queries,
                     inputs=inputs, code_sha256=digest.hexdigest())
    manifest_path = work / 'run_manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest['signature'] != signature:
            raise ValueError('Configuration, input or code changed. Use a NEW --work folder; do not mix artifacts from different runs.')
    else:
        manifest = dict(signature=signature, completed=[], seconds={})
        dump_json(manifest_path, manifest)
    stages = [s for s in STAGES if args.split == 'train' or s != 'label']
    selected = stages if args.stage == 'all' else [args.stage]
    if args.stage not in ['all'] + stages:
        raise ValueError('Test data has no ground-truth labels; omit label stage')
    for stage in selected:
        for prerequisite in stages[:stages.index(stage)]:
            if prerequisite not in manifest['completed']:
                raise ValueError(f'Run --stage {prerequisite} first, or use --stage all')
        if stage in manifest['completed']:
            required = OUTPUTS[stage] + (['tfidf.npz'] if stage == 'index' and args.split == 'train' else [])
            if not all((work / p).exists() for p in required):
                raise ValueError(f'A completed {stage} artifact is missing; use a new work folder')
            print(f'Skip completed stage: {stage}', flush=True)
            continue
        print(f'\nStage: {stage}', flush=True)
        start = time.monotonic()
        if stage == 'index':
            indexing.build(args.cleaned, work, args.split, config)
        elif stage == 'block':
            blocking.generate(work, tfidf, config, args.max_queries)
        elif stage == 'label':
            labeling.label(work, truth_path, config)
        elif stage == 'features':
            features.extract(work, tfidf, args.split, config)
        elif stage == 'export':
            tables.export(work, args.split, config)
        manifest['seconds'][stage] = round(time.monotonic() - start, 3)
        manifest['completed'].append(stage)
        dump_json(manifest_path, manifest)
    print(f'\nOutputs: {work}')


def ensure_complete_test(root):
    work = Path(root) / 'test'
    report = json.loads((work / 'tables_report.json').read_text())
    index = json.loads((work / 'index_report.json').read_text())
    if report['queries'] != index['counts']['S1']:
        raise ValueError('This run contains only a subset of indexed test queries. Prepare ALL test queries before submission.')


def training_arguments(parser):
    parser.add_argument('--model-dir', type=Path, help='Default: WORK/model; choose a new directory for different training settings')
    parser.add_argument('--training-config', type=Path, help='JSON LightGBM/training settings (example: lgbm_config.json)')
    parser.add_argument('--num-boost-round', type=int)
    parser.add_argument('--threads', type=int, help='CPU threads; default 4')
    parser.add_argument('--max-train-pairs', type=int, help='Uniformly sample at most this many training pair rows; default 1000000')


def main():
    parser = argparse.ArgumentParser(description='Cleaning -> blocking -> pair features -> LightGBM -> macro F0.5 -> submission.')
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare', help='Run the connected data preparation modules')
    p.add_argument('--cleaned', type=Path, required=True)
    p.add_argument('--work', type=Path, default=Path('work'))
    p.add_argument('--split', choices=['train', 'test'], default='train')
    p.add_argument('--stage', choices=['all'] + STAGES, default='all')
    p.add_argument('--config', type=Path)
    p.add_argument('--max-queries', type=int, help='Limit S1 queries for a smoke test; reference index still uses ALL S2/S3')
    t = commands.add_parser('train', help='Train LightGBM, score all validation pairs and choose a macro-F0.5 threshold')
    t.add_argument('--work', type=Path, default=Path('work'))
    training_arguments(t)
    pr = commands.add_parser('predict', help='Score every validation/test candidate in batches using the saved model')
    pr.add_argument('--work', type=Path, default=Path('work'))
    pr.add_argument('--split', choices=['validation', 'test'], default='test')
    pr.add_argument('--model-dir', type=Path)
    pr.add_argument('--output', type=Path, help='Optional output path ending in .parquet')
    e = commands.add_parser('evaluate', help='Evaluate validation probabilities with macro F0.5')
    e.add_argument('--work', type=Path, default=Path('work'))
    e.add_argument('--probabilities', type=Path, required=True)
    e.add_argument('--thresholds', default='0.3,0.4,0.5,0.6,0.7,0.8,0.85,0.9,0.925,0.95,0.975,0.99')
    s = commands.add_parser('submit', help='Format test probabilities using an explicit or saved model threshold')
    s.add_argument('--work', type=Path, default=Path('work'))
    s.add_argument('--probabilities', type=Path, required=True)
    s.add_argument('--threshold', type=float, help='Default: threshold selected during training')
    s.add_argument('--model-dir', type=Path)
    b = commands.add_parser('baseline', help='End-to-end: clean if necessary, prepare, train, validate, predict and format output')
    input_group = b.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input', type=Path, help='Raw dataset ZIP/folder; cleaning runs automatically')
    input_group.add_argument('--cleaned', type=Path, help='Already-cleaned dataset folder')
    b.add_argument('--test-cleaned', type=Path, help='Optional separate cleaned folder with full original test data')
    b.add_argument('--work', type=Path, default=Path('work_baseline'))
    b.add_argument('--config', type=Path, help='Blocking/preparation configuration')
    b.add_argument('--skip-test', action='store_true', help='Stop after training and validation')
    training_arguments(b)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args)
    elif args.command == 'train':
        from . import modeling
        cfg = modeling.load_config(args.training_config, dict(num_boost_round=args.num_boost_round,
            num_threads=args.threads, max_train_pairs=args.max_train_pairs))
        modeling.train(args.work, cfg, args.model_dir)
    elif args.command == 'predict':
        from . import modeling
        modeling.predict(args.work, args.split, args.model_dir, args.output)
    elif args.command == 'baseline':
        from .baseline import run
        run(args)
    elif args.command == 'evaluate':
        evaluation.evaluate(args.work / 'train', args.probabilities, [float(t) for t in args.thresholds.split(',')])
    else:
        ensure_complete_test(args.work)
        threshold = args.threshold
        if threshold is None:
            from . import modeling
            _, metadata = modeling.load_model(args.work, args.model_dir)
            provenance = args.probabilities.with_suffix('.metadata.json')
            if not provenance.exists() or json.loads(provenance.read_text()).get('model_sha256') != metadata['model_sha256']:
                raise ValueError('Probabilities do not identify this model. Use predict first, or supply an explicit --threshold for external probabilities.')
            threshold = metadata['decision_threshold']
        evaluation.submit(args.work / 'test', args.probabilities, threshold)
