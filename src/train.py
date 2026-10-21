"""
train.py
--------
Memory-safe streaming training and threshold optimization:
  1. Build blocking index directly from S2/S3 TSVs in a single streaming pass (without storing unused records).
  2. Perform entity-level train/validation split on Source-1 entities.
  3. Generate candidate pairs for train (bounded by MAX_TRAIN_SAMPLES) and validation entities.
  4. Perform targeted single-pass lookup for ONLY the candidate IDs actually needed for model training and validation (~200k IDs instead of 10.3M).
  5. Compute features with guaranteed 1-to-1 label-feature alignment.
  6. Train model variants:
     - class_weight='balanced'
     - class_weight=None
     - moderate positive weighting (scale_pos_weight=2.0)
  7. Sweep decision thresholds on the validation set for Macro F0.5.
  8. Save winning model + threshold + feature list to artifacts.
"""
import argparse
import gc
import json
import os
import random
import time
from collections import defaultdict

import joblib
import numpy as np

from .blocking import BlockingIndex, MAX_CANDIDATES_PER_ENTITY
from .features import build_feature_matrix, FEATURE_NAMES
from .io_utils import parse_id_list

try:
    from lightgbm import LGBMClassifier
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False
    from sklearn.ensemble import GradientBoostingClassifier


TRAIN_BATCH_SIZE = 25_000
MAX_TRAIN_SAMPLES = 900_000
NEGATIVE_TO_POSITIVE_RATIO = 5.0


def make_classifier(weight_mode="balanced", random_state=42):
    if _HAS_LGBM:
        params = {
            "n_estimators": 350,
            "learning_rate": 0.05,
            "max_depth": 6,
            "num_leaves": 31,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_lambda": 1.0,
            "random_state": random_state,
            "n_jobs": -1,
            "verbose": -1,
        }
        if weight_mode == "balanced":
            params["class_weight"] = "balanced"
        elif weight_mode == "moderate":
            params["scale_pos_weight"] = 2.0
        elif weight_mode == "none":
            params["class_weight"] = None
        return LGBMClassifier(**params)

    return GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=4, random_state=random_state
    )


def f_beta(precision, recall, beta=0.5):
    if precision == 0 and recall == 0:
        return 0.0
    b2 = beta * beta
    denom = (b2 * precision) + recall
    if denom == 0:
        return 0.0
    return (1.0 + b2) * precision * recall / denom


def macro_f05_for_threshold(val_s1_ids, val_pairs, probs, ground_truth_map, threshold):
    """val_pairs: list of (s1_id, cand_id) aligned with probs."""
    preds_by_s1 = defaultdict(list)
    for (s1_id, cand_id), p in zip(val_pairs, probs):
        if p >= threshold:
            preds_by_s1[s1_id].append(cand_id)

    scores = []
    for s1_id in val_s1_ids:
        pred_set = set(preds_by_s1.get(s1_id, []))
        true_set = ground_truth_map.get(s1_id, set())
        if not pred_set and not true_set:
            scores.append(1.0)
            continue
        if not pred_set:
            scores.append(0.0)
            continue
        tp = len(pred_set & true_set)
        precision = tp / len(pred_set) if pred_set else 0.0
        recall = tp / len(true_set) if true_set else 0.0
        scores.append(f_beta(precision, recall, beta=0.5))
    return float(np.mean(scores)) if scores else 0.0


def stream_tsv_rows(path):
    """Generator yielding (entity_id, business_name, business_address, country)."""
    with open(path, "r", encoding="utf-8") as f:
        header_line = f.readline()
        if not header_line:
            return
        header = [h.strip() for h in header_line.rstrip("\r\n").split("\t")]
        id_idx = header.index("entity_id") if "entity_id" in header else 0
        name_idx = header.index("business_name") if "business_name" in header else 1
        addr_idx = header.index("business_address") if "business_address" in header else 2
        ctry_idx = header.index("country") if "country" in header else 3

        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) <= max(id_idx, name_idx, addr_idx, ctry_idx):
                continue
            yield parts[id_idx], parts[name_idx], parts[addr_idx], parts[ctry_idx]


def load_ground_truth(path):
    gt_map = {}
    with open(path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if not parts or not parts[0]:
                continue
            s1_id = parts[0]
            matches = parse_id_list(parts[1]) if len(parts) > 1 else []
            gt_map[s1_id] = set(matches)
    return gt_map


def load_targeted_lookup(paths, target_ids):
    """Loads (name, addr) only for the target_ids from the given TSV paths."""
    lookup = {}
    target_set = set(target_ids)
    t0 = time.time()
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            header_line = f.readline()
            header = [h.strip() for h in header_line.rstrip("\r\n").split("\t")]
            id_idx = header.index("entity_id") if "entity_id" in header else 0
            name_idx = header.index("business_name") if "business_name" in header else 1
            addr_idx = header.index("business_address") if "business_address" in header else 2

            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) > max(id_idx, name_idx, addr_idx):
                    rid = parts[id_idx]
                    if rid in target_set:
                        lookup[rid] = (parts[name_idx], parts[addr_idx])
                        if len(lookup) >= len(target_set):
                            break
    print(f"Targeted lookup loaded {len(lookup):,}/{len(target_set):,} requested candidate records in {time.time()-t0:.1f}s.", flush=True)
    return lookup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", default="dataset/train")
    ap.add_argument("--out-dir", default="src/artifacts")
    ap.add_argument("--val-size", type=int, default=20_000,
                    help="Number of Source-1 entities to hold out for local validation.")
    ap.add_argument("--max-train-rows", type=int, default=MAX_TRAIN_SAMPLES,
                    help="Maximum training rows to accumulate for model fitting.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)

    print("=== Step 1: Loading Ground Truth ===", flush=True)
    gt_path = os.path.join(args.train_dir, "train_ground_truth.tsv")
    ground_truth_map = load_ground_truth(gt_path)
    print(f"Loaded ground truth for {len(ground_truth_map):,} Source-1 entities.", flush=True)

    print("\n=== Step 2: Indexing S2 and S3 (Lightweight Single Pass) ===", flush=True)
    s2_path = os.path.join(args.train_dir, "train_source2.tsv")
    s3_path = os.path.join(args.train_dir, "train_source3.tsv")

    index = BlockingIndex(max_candidates=MAX_CANDIDATES_PER_ENTITY)

    print("Indexing Source 2 ...", flush=True)
    index.add_records(stream_tsv_rows(s2_path))
    print("Indexing Source 3 ...", flush=True)
    index.add_records(stream_tsv_rows(s3_path))
    index.finalize()

    print("\n=== Step 3: Partitioning Source-1 Entities ===", flush=True)
    s1_path = os.path.join(args.train_dir, "train_source1.tsv")
    all_s1_records = list(stream_tsv_rows(s1_path))
    total_s1 = len(all_s1_records)
    print(f"Loaded {total_s1:,} Source-1 entities.", flush=True)

    rng = random.Random(args.seed)
    rng.shuffle(all_s1_records)

    n_val = min(args.val_size, int(total_s1 * 0.2))
    val_records = all_s1_records[:n_val]
    # Keep only 60k training entities in memory for batch buffer
    train_records = all_s1_records[n_val : n_val + 60_000]
    val_s1_ids = [r[0] for r in val_records]
    print(f"Entity split: {len(train_records):,} train S1 entities (buffered for sample budget), {len(val_records):,} val S1 entities.")

    del all_s1_records
    gc.collect()

    print("\n=== Step 4: Candidate Generation (Train & Validation) ===", flush=True)
    total_pairs_examined = 0
    total_pos_pairs = 0
    total_neg_pairs = 0
    pos_retained = 0
    neg_retained = 0

    train_pairs_labeled = []
    rng_sample = random.Random(args.seed + 1)
    train_s1_lookup = {}

    for s1_id, name, addr, country in train_records:
        if len(train_pairs_labeled) >= args.max_train_rows:
            break

        train_s1_lookup[s1_id] = (name, addr, country)
        cands = index.candidates_for(name, addr, country)
        true_set = ground_truth_map.get(s1_id, set())

        pos_cands = []
        neg_cands = []
        for c in cands:
            total_pairs_examined += 1
            if c in true_set:
                total_pos_pairs += 1
                pos_cands.append(c)
            else:
                total_neg_pairs += 1
                neg_cands.append(c)

        for c in pos_cands:
            train_pairs_labeled.append((s1_id, c, 1))
            pos_retained += 1

        n_neg_wanted = int(len(pos_cands) * NEGATIVE_TO_POSITIVE_RATIO) if pos_cands else (1 if neg_cands else 0)
        if n_neg_wanted > 0 and neg_cands:
            sampled_negs = rng_sample.sample(neg_cands, min(n_neg_wanted, len(neg_cands)))
            for c in sampled_negs:
                train_pairs_labeled.append((s1_id, c, 0))
                neg_retained += 1

    del train_records
    gc.collect()

    pos_neg_ratio = (neg_retained / pos_retained) if pos_retained else 0.0
    print("\n--- Training Set Extraction Statistics ---")
    print(f"Total candidate pairs examined : {total_pairs_examined:,}")
    print(f"Total positive pairs           : {total_pos_pairs:,}")
    print(f"Total negative pairs           : {total_neg_pairs:,}")
    print(f"Positives retained             : {pos_retained:,}")
    print(f"Negatives retained             : {neg_retained:,}")
    print(f"Final training rows            : {len(train_pairs_labeled):,}")
    print(f"Positive / Negative ratio      : 1 : {pos_neg_ratio:.2f}")

    print("\nGenerating candidates for validation split ...", flush=True)
    val_s1_lookup = {}
    val_pairs = []
    val_total_true = 0
    val_recalled_true = 0
    val_cand_counts = []

    t_val0 = time.time()
    for s1_id, name, addr, country in val_records:
        val_s1_lookup[s1_id] = (name, addr, country)
        cands = index.candidates_for(name, addr, country)
        val_cand_counts.append(len(cands))
        true_set = ground_truth_map.get(s1_id, set())
        val_total_true += len(true_set)
        val_recalled_true += len(true_set & set(cands))

        for c in cands:
            val_pairs.append((s1_id, c))

    del val_records
    gc.collect()

    blocking_recall = val_recalled_true / val_total_true if val_total_true else 0.0
    cand_s = {
        "mean": float(np.mean(val_cand_counts)),
        "p50": float(np.percentile(val_cand_counts, 50)),
        "p90": float(np.percentile(val_cand_counts, 90)),
        "p95": float(np.percentile(val_cand_counts, 95)),
        "p99": float(np.percentile(val_cand_counts, 99)),
        "max": int(max(val_cand_counts)) if val_cand_counts else 0,
    }

    print(f"Validation candidates generated in {time.time()-t_val0:.1f}s")
    print(f"Blocking Recall Ceiling (Validation) : {val_recalled_true:,} / {val_total_true:,} = {blocking_recall*100:.2f}%")
    print(f"Candidates/S1 - Mean: {cand_s['mean']:.2f}, p50: {cand_s['p50']:.1f}, "
          f"p90: {cand_s['p90']:.1f}, p95: {cand_s['p95']:.1f}, p99: {cand_s['p99']:.1f}, max: {cand_s['max']}")

    print("\n=== Step 5: Targeted Record Lookup & Feature Extraction ===", flush=True)
    needed_cand_ids = {c for _, c, _ in train_pairs_labeled} | {c for _, c in val_pairs}
    print(f"Total unique candidate records required: {len(needed_cand_ids):,} (out of 10.3M)", flush=True)

    targeted_lookup = load_targeted_lookup([s2_path, s3_path], needed_cand_ids)

    print("Computing features for training pairs ...", flush=True)
    X_train, kept_train_pairs, y_train = build_feature_matrix(
        train_pairs_labeled, targeted_lookup, s1_lookup=train_s1_lookup
    )
    assert len(X_train) == len(y_train), f"X_train length {len(X_train)} != y_train {len(y_train)}"
    print(f"Training feature matrix ready: {len(X_train):,} rows.", flush=True)

    print("Computing features for validation pairs ...", flush=True)
    X_val, kept_val_pairs = build_feature_matrix(
        val_pairs, targeted_lookup, s1_lookup=val_s1_lookup
    )
    assert len(X_val) == len(kept_val_pairs), f"X_val length {len(X_val)} != kept_val_pairs {len(kept_val_pairs)}"
    print(f"Validation feature matrix ready: {len(X_val):,} rows.", flush=True)

    del targeted_lookup, train_pairs_labeled, val_pairs, train_s1_lookup, val_s1_lookup
    gc.collect()

    print("\n=== Step 6: Model Training & Controlled Weighting Comparison ===", flush=True)
    thresholds_to_test = [0.80, 0.82, 0.84, 0.85, 0.86, 0.87, 0.88, 0.885, 0.89, 0.895,
                          0.90, 0.905, 0.91, 0.915, 0.92, 0.925, 0.93, 0.94, 0.95, 0.96,
                          0.97, 0.98, 0.99, 0.995]
    configs = [
        ("balanced", "class_weight='balanced'"),
        ("none", "No class weighting (default)"),
        ("moderate", "Moderate positive weighting (scale_pos_weight=2.0)"),
    ]

    best_overall_score = -1.0
    best_overall_model = None
    best_overall_thr = 0.94
    best_overall_config_name = "balanced"
    comparison_results = []

    for mode, label in configs:
        print(f"\n--- Testing Configuration: {label} ---", flush=True)
        t_m0 = time.time()
        m = make_classifier(weight_mode=mode, random_state=args.seed)
        m.fit(X_train, y_train)
        print(f"Model fitted in {time.time()-t_m0:.1f}s", flush=True)

        probs = m.predict_proba(X_val)[:, 1] if len(X_val) else np.array([])

        best_cfg_score = -1.0
        best_cfg_thr = 0.94
        for thr in thresholds_to_test:
            score = macro_f05_for_threshold(val_s1_ids, kept_val_pairs, probs, ground_truth_map, thr)
            if score > best_cfg_score:
                best_cfg_score = score
                best_cfg_thr = thr
            print(f"  threshold={thr:.2f} -> val Macro F0.5 = {score:.4f}", flush=True)

        print(f"-> Best for {label}: threshold={best_cfg_thr:.2f}, Macro F0.5={best_cfg_score:.4f}")
        comparison_results.append({
            "mode": mode,
            "label": label,
            "best_threshold": float(best_cfg_thr),
            "val_macro_f05": float(best_cfg_score),
        })

        if best_cfg_score > best_overall_score:
            best_overall_score = best_cfg_score
            best_overall_thr = best_cfg_thr
            best_overall_model = m
            best_overall_config_name = mode

    print("\n=======================================================")
    print("WINNING CONFIGURATION SELECTION (BASED ON HELD-OUT MACRO F0.5)")
    print("=======================================================")
    print(f"Selected Config: {best_overall_config_name}")
    print(f"Selected Threshold: {best_overall_thr:.2f}")
    print(f"Selected Validation Macro F0.5: {best_overall_score:.4f}")

    print("\n=== Step 7: Persisting Model Artifacts ===", flush=True)
    artifact = {
        "model": best_overall_model,
        "threshold": float(best_overall_thr),
        "feature_names": FEATURE_NAMES,
        "backend": "lightgbm" if _HAS_LGBM else "sklearn_gbdt",
        "config_name": best_overall_config_name,
    }
    model_path = os.path.join(args.out_dir, "model.joblib")
    joblib.dump(artifact, model_path)

    report = {
        "n_train_s1_entities": total_s1 - n_val,
        "n_val_s1_entities": n_val,
        "train_candidate_pairs_examined": total_pairs_examined,
        "train_positive_pairs_examined": total_pos_pairs,
        "train_negative_pairs_examined": total_neg_pairs,
        "train_positives_retained": pos_retained,
        "train_negatives_retained": neg_retained,
        "train_final_rows": int(len(y_train)),
        "pos_to_neg_ratio": f"1 : {pos_neg_ratio:.2f}",
        "blocking_recall_ceiling": float(blocking_recall),
        "blocking_stats": cand_s,
        "winning_config": best_overall_config_name,
        "best_threshold": float(best_overall_thr),
        "val_macro_f05": float(best_overall_score),
        "comparison_results": comparison_results,
        "backend": artifact["backend"],
    }
    report_path = os.path.join(args.out_dir, "training_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Artifacts saved to {model_path} and {report_path}.")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except BaseException as e:
        with open("error_train.log", "w", encoding="utf-8") as f_err:
            traceback.print_exc(file=f_err)
        traceback.print_exc()
        raise
