# Amazon ML Challenge 2026 — Complete LightGBM Baseline

**Sampling (optional) → cleaning → blocking → candidate pairs → ground-truth labels → pairwise features → train/validation tables → LightGBM → match probabilities → validation macro F₀.₅ threshold → final matches.**

The repository includes CPU LightGBM training, batched inference, threshold selection, and submission formatting. No challenge datasets, model weights, passwords or generated outputs are committed.

## 1. Install and create your 5,000 + 5,000 sample

Use Python **3.11–3.13**. Open a terminal inside this repository:

```bash
git pull
python -m pip install -r requirements.txt
python make_er_sample.py --input "ML_Dataset.zip" --output "er_sample_5000.zip"
```

The new sampler defaults are:

| Component | Records retained |
|---|---|
| Training Source 1 | **5,000 India + 5,000 US = 10,000 businesses** |
| Training Source 2 / 3 positives | **Every known matching record** for those selected businesses |
| Additional Source 2 / 3 distractors | Up to **5,000 per country in EACH reference source** |
| Test Source 1 / 2 / 3 | Up to **5,000 per country per source**, including France |

Reservoir sampling is uniform within each country and uses seed 42. If a source/country has fewer records than requested, all available records are retained; sampling never creates or duplicates records. Reference files therefore contain more than 10,000 records when all retained positives and distractors are combined. The original ZIP stays unchanged.

You can override the three counts independently:

```bash
python make_er_sample.py --input "ML_Dataset.zip" --output "er_sample_5000.zip" --train-per-country 5000 --distractors-per-country 5000 --test-per-country 5000
```

Choose a new output name if the ZIP already exists. A larger sample is useful for baseline training, but fewer reference distractors still make its validation optimistic. Independently sampled test files do not preserve every true match and **cannot produce a full challenge submission**.

## 2. Run the COMPLETE sampled baseline

```bash
python run.py baseline --input "er_sample_5000.zip" --work "work_5000" --threads 2
```

This single command:

1. Cleans the sampled ZIP into `work_5000/cleaned/`.
2. Builds train indexes, blocks, labels and pairwise features.
3. Keeps all pairs for each S1 in one train/validation split.
4. Trains LightGBM with early stopping.
5. Scores **every validation candidate**, chooses a threshold using macro F₀.₅ over **every validation S1**, and saves the model.
6. Prepares test candidates/features using the frozen training TF-IDF.
7. Scores every test pair in batches and writes `matching_results.tsv` and `candidate_pairs.tsv`.

No GPU or manual probability-file creation is required. In a notebook, prefix terminal commands with `!`.

If you already cleaned the larger sample, use:

```bash
python run.py baseline --cleaned "cleaned_sample_5000" --work "work_5000" --threads 2
```

To train and validate first, without processing test data:

```bash
python run.py baseline --input "er_sample_5000.zip" --work "work_5000" --skip-test --threads 2
```

Re-running the same command skips completed cleaning/preparation/training. Interrupted preparation stages restart from their beginning; interrupted training restarts training. A cleaning interruption requires a new work folder. Changing raw data, preparation configuration or code requires a new work folder. Changing only training settings can use a new `--model-dir`. This avoids silently mixing artifacts.

### Actual challenge submission using ALL original test records

For an initial model trained on your 10,000-S1 sample, but predictions covering the real test dataset:

```bash
python clean_er_data.py --input "ML_Dataset.zip" --output "cleaned_original"
python run.py baseline --cleaned "work_5000/cleaned" --test-cleaned "cleaned_original" --work "work_submission" --threads 2
```

This trains the sampled baseline again in a fresh work folder and searches/predicts against **all original test sources**. Do not replace the sample test inputs inside an existing completed test run.

For preparation and training from the full original training dataset too:

```bash
python run.py baseline --input "ML_Dataset.zip" --work "work_full" --threads 2
```

Full preparation can be very expensive. At a cap of 80 candidates, 2.2 million S1 queries can yield roughly 176 million pairs. The training-row limit below does not reduce indexing, feature-generation or full validation/test-scoring costs.

## 3. Modules and individual commands

| Order | Module | Purpose |
|---|---|---|
| 0 | `make_er_sample.py` | Optional 5,000-per-country sample with positive-match closure |
| 1 | `clean_er_data.py` | Unicode-safe cleaning; preserve raw data and IDs |
| 2 | `src/er_pipeline/indexing.py` | Disk-backed reference indexes and training TF-IDF |
| 3 | `blocking.py` | A–F candidate union, ranking and configurable caps |
| 4 | `labeling.py` | Ground-truth labels, S1-level split and recall audit |
| 5 | `features.py`, `tables.py` | Pairwise features and model-ready tables |
| 6 | **`modeling.py`** | **LightGBM fitting, early stopping, save/load and batched prediction** |
| 7 | `evaluation.py` | Full-query macro F₀.₅ threshold selection and output formatting |
| Orchestration | `baseline.py`, `cli.py` | Connect all stages |

To run each main stage separately:

```bash
python clean_er_data.py --input "er_sample_5000.zip" --output "cleaned_5000"
python run.py prepare --cleaned "cleaned_5000" --work "work_steps" --split train
python run.py train --work "work_steps" --threads 2
python run.py prepare --cleaned "cleaned_5000" --work "work_steps" --split test
python run.py predict --work "work_steps" --split test
python run.py submit --work "work_steps" --probabilities "work_steps/test/test_probabilities.parquet"
```

`train` automatically predicts all validation pairs and evaluates thresholds; `submit` automatically uses that model's saved threshold. For externally generated probabilities, provide `--threshold` explicitly.

Preparation can also be selected module by module with `--stage index`, `block`, `label`, `features`, or `export`, in that order. `prepare` alone still stops at feature tables. Test preparation skips labeling and reuses the training TF-IDF.

A full-reference retrieval pilot can use `prepare --max-queries 1000` in a separate work folder. This limits S1 queries only, not reference indexing; it is not a full-test submission.

## 4. The blocking mechanism

Every block searches S2 AND S3. Known countries restrict retrieval to the same country plus references with unknown country. A query with unknown country searches all countries; it is not dropped. France is included.

| Channel | Keys / retrieval |
|---|---|
| A | Exact basic name, suffix-free Latin-folded name, or sorted name tokens |
| B | First three characters of compact suffix-free name |
| C | State + postal code; city + address token; city + first three name tokens |
| D | Zero-normalized house number + name prefix |
| E | Name words and within-word character trigrams → indexed shortlist → hashed TF-IDF cosine reranking |
| F | Address words and character trigrams → indexed shortlist → address TF-IDF reranking |

Country applies to every channel. Missing key components do not become shared empty-string blocks. B is only a prefix heuristic; it is NOT a substitute for E. Whole-word reordering is handled by sorted-name A, bag-of-word/within-word grams E, and pair features.

Candidate generation:

1. Retrieve bounded candidates from each channel.
2. Union and deduplicate by record ID. Identical text with different IDs remains separate.
3. Rank by `0.55 × name cosine + 0.30 × address cosine + 0.10 × exact basic name + 0.01 × channel count`.
4. Apply the configurable final cap. The result is the ACTUAL set passed to feature extraction and the model.

Missing similarities contribute zero **only to this retrieval ranking heuristic**. Their model feature values remain NaN, with missingness flags.

Broad-block protection: if an exact/prefix/component block exceeds `block_limit`, it is searched using lexical relevance within that block. It is never accepted as an unbounded Cartesian product. Overflow and final-cap counts are reported. Exact-name blocks can also be capped; inspect recall before choosing limits.

E/F are **approximate retrieval**, not exhaustive global TF-IDF nearest neighbours. SQLite FTS5 BM25 finds a bounded shortlist and TF-IDF reranks it. Some true neighbours may be outside that shortlist. Compare candidate recall while adjusting `lexical_pool`, `query_terms`, `lexical_top_k` and `max_candidates`.

## 5. Ground-truth labels and validation

For each actual candidate `(Source 1 ID, candidate ID)`:

- `label = 1` if the ID occurs in that S1 ground-truth list.
- `label = 0` otherwise.

All generated negatives are retained; there is no hidden negative downsampling. Known positive pairs missed by blocking are **not inserted** into candidate/validation tables. Missing ground-truth rows or referenced records are input errors, not assumed negatives.

Each S1 ID is deterministically assigned to approximately 80% train / 20% validation using a seeded hash including country and singleton status. This is approximate stratification, not an exact quota per stratum. **All pairs belonging to an S1 stay together.** Check group counts, especially on tiny samples. Train and validation retrieve against the same full training S2/S3 pool, matching the reference-search task. This is an S1 holdout, not a guarantee that every business name/token is unseen.

TF-IDF statistics use only training reference (S2/S3) text. They never use labels, held-out S1 text or test text. No business identity lookup, geocoding or external data is used.

`queries.tsv.gz` includes every selected S1, even with zero candidates. `query_labels.tsv.gz` additionally contains the split, full truth and hit counts. Keep these files: pair tables cannot represent zero-candidate queries on their own.

`candidate_recall.json` reports overall, train/validation and country-specific:

- Candidate recall = retrieved true links / ALL true links.
- Fraction of non-singletons with all true matches retrieved.
- Positive and negative pair counts.
- Oracle macro F₀.₅ ceiling: best possible score if every retrieved true match is selected and every false candidate rejected. **This is not a trained model score.**

## 6. Pairwise features

Use the exact `features` list in `feature_columns.json` for model training. Exclude IDs, `dataset_split` and `label` from model inputs. `label` is your target.

| Features | Definition |
|---|---|
| `name_levenshtein_ratio`, `address_levenshtein` | Normalized Levenshtein similarity: 1 − edit distance / maximum length |
| `name_jaro_winkler` | RapidFuzz normalized Jaro–Winkler similarity |
| `name_token_jaccard`, `address_token_jaccard` | Intersection / union of token sets |
| `name_tfidf_cosine`, `address_tfidf_cosine` | Cosine of frozen hashed TF-IDF vectors |
| `name_char_ngram_similarity`, `address_char_similarity` | Jaccard of word-boundary character trigram sets |
| `name_exact`, `address_exact` | Equality of nonempty normalized strings |
| `name_without_suffix_similarity`, `name_expanded_similarity` | Normalized Levenshtein on those separate name views |
| `postal_code_exact`, `city_exact`, `state_exact`, `house_number_exact`, `country_exact` | Equality when both components are present |
| `name_address_similarity_product` | Name Levenshtein similarity × address Levenshtein similarity |
| `name_missing`, `address_missing` | 1 if either side is missing |
| `*_missing_left`, `*_missing_right`, `*_missing_both` | Separate missingness indicators |
| Component missing flags | 1 if either component is absent |
| `address_edit_distance` | Raw edit distance, unlike the normalized similarity column |
| Name/address length ratios | Shorter length / longer length |
| Retrieval score/rank/source and block flags | Additional label-free retrieval evidence |

Similarities/equalities involving missing text/components are **NaN**, even when both are missing. Two empty addresses are not an exact-address match. Do not replace these NaNs with zero.

TF-IDF uses word tokens plus within-word 3–5-grams, log term frequency `1 + log(tf)`, smoothed IDF `log((N+1)/(df+1))+1`, and L2 normalization. Names use the suffix-free Latin-folded view with a basic-name fallback; addresses use the normalized Latin-folded view. Hashing keeps IDF memory fixed; collisions are possible, so these cosines approximate an explicit-vocabulary TF-IDF representation. Retrieval token encoding itself is lossless UTF-8 hex, not hashed.

Unicode is retained. Character grams do not translate Hindi/Tamil/etc. into Latin. Cross-script matches may require address evidence or a later multilingual retrieval channel. The cleaner's city/state/postal/house columns remain heuristic evidence. House-number disagreement never automatically rejects a match.

## 7. LightGBM settings and memory use

`lgbm_config.json` is the editable training configuration. `config.json` separately controls blocking/preparation. Pass them using `--training-config` and `--config` respectively.

```bash
python run.py train --work "work_5000" --training-config "lgbm_config.json" --model-dir "work_5000/model_v2"
```

| Training setting | Default |
|---|---:|
| Objective / device | Binary classification / CPU |
| Learning rate | 0.05 |
| Maximum boosting rounds | 600 |
| Early-stopping patience | 50 |
| Leaves / minimum rows per leaf | 31 / 50 |
| Maximum bins | 63 |
| L2 regularization | 2.0 |
| CPU threads | 4; use `--threads 2` on a small laptop |
| Maximum training pairs | 1,000,000 |
| Maximum early-stopping monitor pairs | 200,000 |
| Read/prediction batch size | 20,000 |

Training and early-stopping arrays use float32. When a table exceeds its cap, the loader takes a reproducible uniform sample of pair rows without replacement. It does not move rows between train and validation, rebalance classes or inject positives. The cap preserves class proportions in expectation; exact class counts are recorded in model metadata. **All validation pairs are still scored for final threshold selection**, including those outside the early-stopping monitor sample. Test pairs are never downsampled during prediction.

The requested 10,000 training S1 records remain in preparation, with approximately 20% held out for validation. At an 80-candidate cap, that sample fits below the default one-million-training-pair limit, so all its generated training pairs are normally used.

The loader bounds raw training arrays, but LightGBM requires extra memory for bins, trees and working buffers. This is not fully out-of-core training. If RAM is tight, lower `--max-train-pairs`, lower `max_early_stopping_pairs` in the training config, and use fewer threads. Indexes and feature tables still consume disk space.

NaN features are passed directly to LightGBM; they are not filled with zero. IDs, labels and split names are excluded from model inputs. No class weighting is enabled by default. Predicted probabilities are binary-model outputs, not a guarantee of calibration on a different reference pool.

Early stopping optimizes **binary log loss** on the validation monitor. After training, the saved best iteration scores the whole validation candidate table. A threshold grid is evaluated using the challenge's **macro F₀.₅**:

`F0.5 = 5*TP / (5*TP + 4*FP + FN)` per Source 1, then averaged.

Both truth and prediction empty → 1; true singleton with any predicted match → 0. Blocked-out positives and zero-candidate queries remain in the denominator. A threshold tie chooses the higher threshold. Validation is used for stopping and threshold selection, so this score is a tuning result, not an unbiased final-test estimate. The model is not automatically refit on held-out validation data after tuning.

To try more thresholds after training:

```bash
python run.py evaluate --work "work_5000" --probabilities "work_5000/model/validation_probabilities.parquet" --thresholds "0.4,0.45,0.5,0.55,0.6,0.7,0.8,0.9,0.95,0.99"
```

This separate evaluation does not overwrite the threshold bound to the saved model. To use its new threshold, pass `submit --threshold YOUR_VALUE` explicitly or retrain with the revised grid in a new model directory.

## 8. Outputs and submission checks

| File under your work folder | Contents |
|---|---|
| `model/lightgbm_model.txt` | Saved LightGBM model |
| `model/model_metadata.json` | Feature order, settings, row counts, model hash, best iteration, selected threshold and validation score |
| `model/validation_probabilities.parquet` | Every validation candidate probability |
| `model/validation_f05.json` | Threshold scores for this model |
| `model/feature_importance.tsv` | Gain/split feature importance |
| `model/learning_curve.json` | Validation log-loss history |
| `train/train_features.parquet`, `train/validation_features.parquet` | Labeled model tables |
| `train/candidate_recall.json` | Retrieval recall and its oracle score ceiling |
| `test/test_features.parquet` | Unlabeled test model inputs |
| `test/test_probabilities.parquet` | Every test candidate probability |
| **`test/matching_results.tsv`** | One row per indexed S1 with zero, one or multiple accepted matches |
| **`test/candidate_pairs.tsv`** | The exact final candidate set scored by the model |
| `baseline_result.json` | End-to-end result summary |

`feature_columns.json` records the exact feature list. `queries.tsv.gz` and `query_labels.tsv.gz` retain queries with no candidates. Model hashes/feature order and frozen TF-IDF are checked before loading/predicting. Formatting rejects missing, duplicate or unexpected probability rows and incomplete indexed test-query runs.

Before real submission, validate both TSVs against the **ORIGINAL full test sources**, not the sampled test files:

```bash
python utils/validate_submission.py --matching "work_submission/test/matching_results.tsv" --candidate "work_submission/test/candidate_pairs.tsv" --test-dir "dataset/test" --check-ids
```

`utils/validate_submission.py` comes from your official challenge resource bundle; adjust its path accordingly. Its optional ID check may require significant RAM on full reference files. Successful validation against a small sampled test folder proves only sample formatting.

## 9. Blocking controls and verification

Default blocking settings in `config.json`: `max_candidates=80`, `block_limit=100`, `lexical_pool=120`, `lexical_top_k=25`, `query_terms=16`, `hash_bits=18`, `row_group_size=10000`, `sqlite_cache_mb=128`. Lower candidate caps reduce cost but may reduce recall. Defaults are starting values, not tuned competition optima.

SQLite must include FTS5. Most standard Python builds include it; an unavailable extension raises `no such module: fts5` during indexing.

Run tests:

```bash
python -m unittest discover -s tests -v
```

Tests cover 5,000-per-country sampling, connected preparation, country restrictions, missing evidence, duplicate-looking record IDs, S1 split separation, actual LightGBM training/reload, deterministic capped loading, all-candidate probability coverage, empty prediction tables, feature/threshold consistency and macro F₀.₅ including missed positives.

`SAMPLE_VERIFICATION.md` records the original preparation checks and the subsequent end-to-end LightGBM smoke test. The full 1.01 GB dataset and new 10,000-S1 sample have not been run in this workspace because only the earlier small sample was uploaded.

Official implementation references: [LightGBM 4.6 train](https://lightgbm.readthedocs.io/en/v4.6.0/pythonapi/lightgbm.train.html), [LightGBM Booster](https://lightgbm.readthedocs.io/en/v4.6.0/pythonapi/lightgbm.Booster.html), [SQLite FTS5](https://www.sqlite.org/fts5.html), [RapidFuzz distances](https://rapidfuzz.github.io/RapidFuzz/Usage/distance/Levenshtein.html), [PyArrow ParquetWriter](https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetWriter.html).
