# Uploaded-sample verification

Tested with Python 3.12, NumPy 2.3.5, RapidFuzz 3.14.3 and PyArrow 23.0.1.
The original uploaded `er_sample.zip` was first processed by your cleaning script.
No model was trained during that original preparation-only check.

| Check | Observed result |
|---|---:|
| Training Source 1 queries | 500 |
| Training Source 2 references | 1,384 |
| Training Source 3 references | 1,437 |
| Known true links | 1,821 |
| True links in final candidate sets | 1,811 |
| Candidate recall | 99.4509% |
| Final training-dataset candidate pairs | 24,845 |
| Mean candidates per training S1 | 49.69 |
| Model-training S1 / validation S1 | 397 / 103 |
| Model-training pair rows | 19,814 |
| Validation pair rows | 5,031 |
| Test queries retained | 300 |
| France test queries retained | 100 |
| Test pair rows | 10,985 |

Every generated training/validation label was compared with the sample ground-truth table. Source 1 IDs were disjoint between train and validation. All 514 sampled training/validation pairs with missing addresses had NaN address similarity/equality features.

Synthetic tests additionally checked zero-candidate queries, true matches deliberately missed by blocking, singleton scoring, duplicate-looking records with distinct IDs, country restrictions, overflowing blocks, token reordering, and missing-probability rejection.

For metric verification ONLY, ground-truth-based probabilities were fed into the evaluator. Its validation score exactly matched the independently calculated retrieval oracle ceiling. **That value is not a model result and is deliberately not advertised as performance.**

For submission-format verification ONLY, zero-probability test fixtures produced empty predictions for all 300 S1 records. Both files passed the challenge-provided validator with `--check-ids`. The project ZIP does not include these dummy probabilities or predictions.

Rerunning the preparation command correctly skipped completed stages.

**Limitations:** The uploaded sample contains relatively few distractors and its train positives were intentionally preserved during sampling. Its high recall does not predict full-dataset recall. Independent test source samples do not preserve all matching relationships, so test recall was not measured. Full-data runtime, disk size, trained F₀.₅ and leaderboard performance have not been measured.


## LightGBM end-to-end check (2026-09-25)

The new `baseline --input` command ran from the original uploaded small ZIP through cleaning, preparation, actual LightGBM training, validation prediction/threshold selection, all sample test predictions and submission formatting. This used 80 boosting rounds and 2 CPU threads, not the default maximum of 600 rounds.

- Training pairs: 19,814; validation pairs scored: 5,031.
- Validation Source 1 records: 103.
- Selected threshold: 0.55; tuned sample macro F0.5: 0.986578.
- Every one of the 10,985 sample test pairs received a probability.
- Both output TSVs contain all 300 sample test queries, including France.

This is a **small-sample smoke-test result, not a leaderboard estimate**. The full original data and requested new 10,000-S1 sample were not available for training here. Independent reservoir tests verified exactly 5,000 selected records each from India and the US when supplied with 6,200 records per country.

The actual LightGBM-generated sample output passed the official submission validator with `--check-ids` (300 S1 rows, 45 nonempty predictions). All six unit/integration tests passed. Re-running the baseline skipped completed cleaning, preparation and training, then regenerated predictions successfully.
