# Business entity resolution: CPU training on PBS HPC

Match each Source 1 business against zero or more Source 2 and 3 records. This repository contains the CPU LightGBM pipeline, configs, tests, and a PBS job. Keep challenge data and output outside Git.

## Set up on the HPC login node

The provided cluster example uses PBS and queue `workq`. Check limits with `qstat -Qf workq`, then adjust `hpc/train_cpu.pbs` if 16 CPUs, 48 GB RAM, or 24 hours is disallowed. The login node's 64 logical CPUs and 124 GiB RAM are shared, not your compute job allocation. No GPU is requested.

```bash
git clone https://github.com/raoankit72005/ML_Code_Phase1.git
cd ML_Code_Phase1
module load anaconda3  # if available; otherwise use cluster Python 3.11 or 3.12
python3 -m venv "$HOME/er-venv"
source "$HOME/er-venv/bin/activate"
python -m pip install -r requirements.txt
```

Put the complete original `ML_Dataset.zip` in HPC home/project storage. Make a representative training sample while preserving positive matches:

```bash
python make_er_sample.py --input "$HOME/ML_Dataset.zip" --output "$HOME/er_train_sample.zip" \
  --train-per-country 5000 --distractors-per-country 5000 --test-per-country 5000
```

This sample trains on up to 10,000 S1 entities across India and the US. The job ignores its test subset and scores **every original test S1 record**. Sampling speeds training but may reduce accuracy. Processing millions of original test records can still take many hours and significant disk space; one-hour completion is not guaranteed.

## Submit and monitor

```bash
export DATA_ZIP="$HOME/ML_Dataset.zip"
export SAMPLE_ZIP="$HOME/er_train_sample.zip"
export RUN_DIR="$HOME/er_run_01"
export VENV_DIR="$HOME/er-venv"
qsub -v DATA_ZIP,SAMPLE_ZIP,RUN_DIR,VENV_DIR hpc/train_cpu.pbs
qstat -u "$USER"
```

PBS writes a combined log `er-cpu.o<job-id>` in the repository directory. Watch it with `tail -f er-cpu.o<job-id>`. After completion:

```bash
ls -lh "$RUN_DIR/experiment/model/lightgbm_model.txt" \
  "$RUN_DIR/experiment/test/matching_results.tsv" \
  "$RUN_DIR/experiment/test/candidate_pairs.tsv"
cat "$RUN_DIR/experiment/baseline_result.json"
```

Upload `matching_results.tsv` to the challenge portal. Keep `candidate_pairs.tsv` for the final package. Validate both against the **original test sources** using `utils/validate_submission.py` from the official challenge bundle (options: `--matching`, `--candidate`, `--test-dir`, `--check-ids`). The validator may use substantial RAM for full ID checking.

Completed preparation stages can be skipped when rerunning the same command. An interrupted cleaning step or a change to data, code, or preparation config needs a new `RUN_DIR`. The full original test set may create many candidate pairs (the default cap is 80 per S1) and require substantial disk space.

## Local sample smoke test

```bash
python -m unittest discover -s tests -v
python run.py baseline --input '/path/to/er_sample.zip' --work "$HOME/er_smoke" --threads 4
```

The small sample checks execution but is not a complete submission. Add `--skip-test` for a training-only pilot. The final model is in `experiment/model/`; validation metrics are in `experiment/model/validation_f05.json`.
