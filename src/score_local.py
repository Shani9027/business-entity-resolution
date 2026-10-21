"""
score_local.py
---------------
Computes the exact macro-averaged F_0.5 metric described in the challenge,
given a matching_results.tsv and a ground-truth file with the same two-column
schema. Use this on a train-derived hold-out split (there is no ground truth
for the real test set) to estimate your leaderboard score before submitting.

Usage:
    python -m src.score_local --predictions output/matching_results.tsv --ground-truth path/to/gt.tsv
"""
import argparse
import pandas as pd
from .io_utils import parse_id_list


def f_beta(precision, recall, beta=0.5):
    if precision == 0 and recall == 0:
        return 0.0
    b2 = beta * beta
    denom = (b2 * precision) + recall
    if denom == 0:
        return 0.0
    return (1 + b2) * precision * recall / denom


def score(predictions_path, ground_truth_path):
    pred = pd.read_csv(predictions_path, sep="\t", dtype=str, keep_default_na=False)
    gt = pd.read_csv(ground_truth_path, sep="\t", dtype=str, keep_default_na=False)

    pred_map = {r.source1_entity_id: set(parse_id_list(r.matched_entity_ids))
                for r in pred.itertuples(index=False)}
    gt_map = {r.source1_entity_id: set(parse_id_list(r.matched_entity_ids))
              for r in gt.itertuples(index=False)}

    scores = []
    n_singleton_correct, n_singleton_total = 0, 0
    for s1_id, true_set in gt_map.items():
        pred_set = pred_map.get(s1_id, set())
        if not true_set:
            n_singleton_total += 1
            if not pred_set:
                n_singleton_correct += 1
        if not pred_set and not true_set:
            scores.append(1.0)
            continue
        if not pred_set:
            scores.append(0.0)
            continue
        tp = len(pred_set & true_set)
        precision = tp / len(pred_set)
        recall = tp / len(true_set) if true_set else 0.0
        scores.append(f_beta(precision, recall, beta=0.5))

    macro_f05 = sum(scores) / len(scores) if scores else 0.0
    print(f"Evaluated {len(scores)} Source-1 entities.")
    print(f"Singleton accuracy: {n_singleton_correct}/{n_singleton_total}")
    print(f"Macro F_0.5: {macro_f05:.4f}")
    return macro_f05


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--ground-truth", required=True)
    args = ap.parse_args()
    score(args.predictions, args.ground_truth)
