"""Evaluate model probabilities with per-Source-1 macro F0.5."""
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from .common import connect, rows, parquet_rows, dump_json, truth_ids


def f05(truth, predictions):
    if not truth and not predictions:
        return 1.0
    tp = len(truth & predictions)
    fp, fn = len(predictions - truth), len(truth - predictions)
    denominator = 5 * tp + 4 * fp + fn
    return 5 * tp / denominator if denominator else 0.0


def probability_store(work, probabilities, validation):
    path = work / ('validation_probabilities.sqlite' if validation else 'test_probabilities.sqlite')
    if path.exists():
        path.unlink()  # This module's generated scratch database only.
    conn = connect(path)
    conn.execute('CREATE TABLE scores(sid TEXT, eid TEXT, probability REAL, PRIMARY KEY(sid,eid)) WITHOUT ROWID')
    source = work / ('validation_features.parquet' if validation else 'test_features.parquet')
    n = 0
    for pair in parquet_rows(source):
        conn.execute('INSERT INTO scores VALUES(?,?,NULL)', (pair['source1_entity_id'], pair['candidate_entity_id']))
        n += 1
        if n % 10000 == 0:
            conn.commit()
    conn.commit()
    reader = parquet_rows(probabilities) if str(probabilities).endswith('.parquet') else rows(probabilities)
    count = 0
    for row in reader:
        p = float(row['match_probability'])
        if not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError('match_probability must be finite and in [0,1]')
        cur = conn.execute('UPDATE scores SET probability=? WHERE sid=? AND eid=? AND probability IS NULL',
            (p, row['source1_entity_id'], row['candidate_entity_id']))
        if cur.rowcount != 1:
            raise ValueError('Duplicate or unexpected probability pair: ' + str(row))
        count += 1
        if count % 10000 == 0:
            conn.commit()
    conn.commit()
    if count != n:
        raise ValueError(f'Expected {n} probabilities, received {count}; score EVERY final candidate.')
    return conn


def evaluate(work, probabilities, thresholds):
    if not thresholds or any(not math.isfinite(t) or not 0 <= t <= 1 for t in thresholds):
        raise ValueError('Thresholds must be in [0,1]')
    thresholds = sorted(set(thresholds))
    conn = probability_store(work, probabilities, validation=True)
    sums = {t: defaultdict(float) for t in thresholds}
    counts = defaultdict(int)
    for query in rows(work / 'query_labels.tsv.gz'):
        if query['dataset_split'] != 'validation':
            continue
        sid, country = query['source1_entity_id'], query['country']
        truth = set(truth_ids(query['matched_entity_ids']))
        pairs = list(conn.execute('SELECT eid,probability FROM scores WHERE sid=?', (sid,)))
        for threshold in thresholds:
            predicted = {eid for eid, p in pairs if p >= threshold}
            value = f05(truth, predicted)
            sums[threshold]['all'] += value
            sums[threshold]['country:' + country] += value
        counts['all'] += 1
        counts['country:' + country] += 1
    conn.close()
    if not counts['all']:
        raise ValueError('No validation queries; increase the query sample or validation fraction')
    results = [{'threshold': t, 'macro_f05': sums[t]['all'] / counts['all'],
                'by_country': {k: sums[t][k] / n for k, n in counts.items() if k != 'all'}} for t in thresholds]
    best = max(results, key=lambda r: (r['macro_f05'], r['threshold']))
    result = dict(best=best, results=results, query_counts=dict(counts),
        note='Macro per S1, including singletons, zero-candidate queries and blocked-out positives. Threshold selected on validation; this is not an unbiased final-test estimate.')
    dump_json(work / 'validation_f05.json', result)
    print(json.dumps(best, indent=2))
    return result


def submit(work, probabilities, threshold):
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Threshold must be in [0,1]')
    conn = probability_store(work, probabilities, validation=False)
    with (work / 'matching_results.tsv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t', lineterminator='\n')
        writer.writerow(['source1_entity_id', 'matched_entity_ids'])
        for query in rows(work / 'queries.tsv.gz'):
            sid = query['source1_entity_id']
            ids = [r[0] for r in conn.execute('SELECT eid FROM scores WHERE sid=? AND probability>=? ORDER BY eid', (sid, threshold))]
            writer.writerow([sid, ','.join(ids)])
    conn.close()
    print('matching_results.tsv created. Run the challenge-provided submission validator before uploading.')
