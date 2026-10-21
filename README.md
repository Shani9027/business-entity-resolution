# Business Entity Resolution — Pipeline

End-to-end pipeline for the Amazon ML Challenge 2026 "Business Entity
Resolution" problem: matches Source-2 / Source-3 business records to the
deduplicated Source-1 reference list.

## Directory layout

```
business_entity_resolution/
├── src/
│   ├── normalize.py       # name/address text normalization
│   ├── blocking.py        # candidate generation (inverted-index blocking)
│   ├── features.py        # similarity feature engineering
│   ├── io_utils.py         # TSV read/write helpers
│   ├── train.py            # trains the classifier + tunes the threshold
│   ├── infer.py             # runs blocking + model on the test set, writes outputs
│   ├── score_local.py      # computes macro F_0.5 against a held-out ground truth
│   └── artifacts/           # trained model.joblib + training_report.json (created by train.py)
├── requirements.txt
└── README.md
```

`utils/validate_submission.py` (one level up, alongside `output/`) is a local
re-implementation of the organizers' validator, for use before the official
script is available.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Data layout expected

Place the real challenge data (unzipped) as:

```
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

(Run these commands from the `business_entity_resolution/` folder, i.e. the
folder this README is in — the default paths below are relative to it. Adjust
`--train-dir` / `--test-dir` if your data lives elsewhere.)

## 1. Train

```bash
python3 -m src.train --train-dir dataset/train --out-dir src/artifacts
```

This will:
- Split Source-1 entities 80/20 into train/val (entity-level split, no leakage)
- Build the blocking index over Source-2/Source-3 training records
- Generate candidate pairs for both splits and print a **blocking recall
  ceiling** diagnostic (fraction of true matches the blocking stage actually
  retrieves — this is the upper bound on achievable recall)
- Extract similarity features for every candidate pair
- Train a LightGBM classifier (falls back automatically to scikit-learn's
  `GradientBoostingClassifier` if LightGBM isn't installed)
- Sweep the decision threshold on the validation split to maximize **macro
  F_0.5** (the actual competition metric)
- Save `src/artifacts/model.joblib` (model + threshold + feature list) and
  `src/artifacts/training_report.json` (summary metrics)

## 2. Run inference on the test set

```bash
python3 -m src.infer --test-dir dataset/test --model-dir src/artifacts --out-dir output
```

Writes `output/candidate_pairs.tsv` and `output/matching_results.tsv` in the
exact required format. Every Source-1 test entity — including any new
country such as France — gets exactly one row in both files.

## 3. (Optional) Score a hold-out split locally

Since the real test set has no released ground truth, evaluate on a held-out
slice of the training data instead, e.g. by running `train.py`/`infer.py`
against a train/train-split you create yourself, then:

```bash
python3 -m src.score_local --predictions output/matching_results.tsv --ground-truth path/to/your_holdout_gt.tsv
```

This reports the exact macro-averaged F_0.5 metric used by the leaderboard.

## 4. Validate the output format before submitting

```bash
python3 ../../../utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

(Path shown relative to this folder inside the submission zip — adjust to
wherever `utils/validate_submission.py` actually sits, or use the
organizers' official copy once released.)

## Design notes

- **No external data / lookups** are used anywhere in the pipeline (fair-play
  compliant) — every feature is derived purely from `business_name`,
  `business_address`, and `country` as provided.
- **`country` is treated as an open string label** throughout — never
  hardcoded to `{US, India}` — so an unseen label like `France` at test time
  is handled automatically by the same code path.
- **Model size**: LightGBM gradient-boosted trees, MIT-licensed, a few
  hundred KB on disk — far under the 8B-parameter ceiling.
- See `Documentation_template.md` (in the submission zip root) for the full
  methodology write-up, blocking-quality analysis, and feature list.
