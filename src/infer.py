"""
infer.py
--------
Memory-safe streaming test inference:
  1. Stream test_source2.tsv and test_source3.tsv to build the blocking index and candidate lookup.
  2. Stream test_source1.tsv in batches (TEST_BATCH_SIZE).
  3. For each batch:
     - Generate candidates
     - Write candidate_pairs.tsv incrementally
     - Compute features
     - Predict match probabilities
     - Filter pairs with score >= threshold
     - Write matching_results.tsv incrementally
     - Release chunk structures and reclaim memory
  4. Guarantees:
     - matched_entity_ids is always a strict subset of candidate_entity_ids
     - Exactly one row per test Source-1 entity
     - Correct TSV headers and no invalid/duplicate IDs
     - Bounded memory footprint throughout execution
"""
import argparse
import gc
import os
import time
from collections import defaultdict

import joblib
import numpy as np

from .blocking import BlockingIndex, MAX_CANDIDATES_PER_ENTITY
from .features import build_feature_matrix
from .train import stream_tsv_rows

TEST_BATCH_SIZE = 25_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", default="dataset/test")
    ap.add_argument("--model-dir", default="src/artifacts")
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Override decision threshold stored in model artifact.")
    ap.add_argument("--batch-size", type=int, default=TEST_BATCH_SIZE)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    t0_start = time.time()

    print("=== Step 1: Loading Trained Model ===", flush=True)
    model_path = os.path.join(args.model_dir, "model.joblib")
    artifact = joblib.load(model_path)
    model = artifact["model"]
    threshold = args.threshold if args.threshold is not None else artifact["threshold"]
    print(f"Loaded model backend={artifact.get('backend', 'unknown')}, "
          f"config={artifact.get('config_name', 'default')}, "
          f"decision threshold={threshold:.3f}", flush=True)

    print("\n=== Step 2: Indexing Test S2 & S3 (Single Streaming Pass) ===", flush=True)
    s2_path = os.path.join(args.test_dir, "test_source2.tsv")
    s3_path = os.path.join(args.test_dir, "test_source3.tsv")

    index = BlockingIndex(max_candidates=MAX_CANDIDATES_PER_ENTITY)
    s2_s3_lookup = {}

    print("Indexing test Source 2 ...", flush=True)
    index.add_records(stream_tsv_rows(s2_path), record_lookup=s2_s3_lookup)
    print("Indexing test Source 3 ...", flush=True)
    index.add_records(stream_tsv_rows(s3_path), record_lookup=s2_s3_lookup)
    index.finalize()
    print(f"Total test candidate pool records: {len(s2_s3_lookup):,}", flush=True)

    print("\n=== Step 3: Streaming Test Inference & Writing Outputs ===", flush=True)
    s1_path = os.path.join(args.test_dir, "test_source1.tsv")

    cand_path = os.path.join(args.out_dir, "candidate_pairs.tsv")
    match_path = os.path.join(args.out_dir, "matching_results.tsv")

    f_cand = open(cand_path, "w", encoding="utf-8", newline="")
    f_match = open(match_path, "w", encoding="utf-8", newline="")

    f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
    f_match.write("source1_entity_id\tmatched_entity_ids\n")

    total_s1_processed = 0
    total_candidates_generated = 0
    total_matches_predicted = 0
    singletons_predicted = 0
    entities_with_matches = 0

    batch = []
    t_inf0 = time.time()

    for s1_row in stream_tsv_rows(s1_path):
        batch.append(s1_row)
        if len(batch) >= args.batch_size:
            # Process batch
            batch_s1_lookup = {r[0]: (r[1], r[2], r[3]) for r in batch}
            batch_cands = {}
            pairs_to_score = []

            for s1_id, name, addr, country in batch:
                cands = index.candidates_for(name, addr, country)
                # Deduplicate while preserving rank order
                seen = set()
                uniq_cands = [c for c in cands if not (c in seen or seen.add(c))]
                batch_cands[s1_id] = uniq_cands
                total_candidates_generated += len(uniq_cands)

                # Write candidate_pairs.tsv row immediately
                f_cand.write(f"{s1_id}\t{','.join(uniq_cands)}\n")

                for c in uniq_cands:
                    pairs_to_score.append((s1_id, c))

            # Run ML model on candidate pairs for this batch
            batch_matches = defaultdict(list)
            if pairs_to_score:
                X_b, kept_b = build_feature_matrix(pairs_to_score, s2_s3_lookup, s1_lookup=batch_s1_lookup)
                if len(X_b) > 0:
                    probs = model.predict_proba(X_b)[:, 1]
                    for (s1_id, cand_id), p in zip(kept_b, probs):
                        if p >= threshold:
                            batch_matches[s1_id].append((cand_id, p))

            # Write matching_results.tsv rows for this batch
            for s1_id, _, _, _ in batch:
                cand_list = batch_cands[s1_id]
                cand_set = set(cand_list)

                # Rank matches by predicted probability descending
                raw_matches = batch_matches.get(s1_id, [])
                ranked_matches = [cid for cid, _ in sorted(raw_matches, key=lambda t: t[1], reverse=True)]

                # Enforce final_matches ⊆ candidate_pairs and uniqueness
                seen_m = set()
                final_matches = [m for m in ranked_matches if m in cand_set and not (m in seen_m or seen_m.add(m))]

                if final_matches:
                    entities_with_matches += 1
                    total_matches_predicted += len(final_matches)
                else:
                    singletons_predicted += 1

                f_match.write(f"{s1_id}\t{','.join(final_matches)}\n")

            total_s1_processed += len(batch)
            elapsed = time.time() - t_inf0
            rate = total_s1_processed / elapsed if elapsed else 0.0
            print(f"  [infer] processed {total_s1_processed:,} test S1 entities "
                  f"({rate:,.0f} entities/sec, {entities_with_matches:,} matched)", flush=True)

            del batch, batch_s1_lookup, batch_cands, pairs_to_score, batch_matches
            batch = []
            gc.collect()

    # Process final partial batch if any
    if batch:
        batch_s1_lookup = {r[0]: (r[1], r[2], r[3]) for r in batch}
        batch_cands = {}
        pairs_to_score = []

        for s1_id, name, addr, country in batch:
            cands = index.candidates_for(name, addr, country)
            seen = set()
            uniq_cands = [c for c in cands if not (c in seen or seen.add(c))]
            batch_cands[s1_id] = uniq_cands
            total_candidates_generated += len(uniq_cands)

            f_cand.write(f"{s1_id}\t{','.join(uniq_cands)}\n")

            for c in uniq_cands:
                pairs_to_score.append((s1_id, c))

        batch_matches = defaultdict(list)
        if pairs_to_score:
            X_b, kept_b = build_feature_matrix(pairs_to_score, s2_s3_lookup, s1_lookup=batch_s1_lookup)
            if len(X_b) > 0:
                probs = model.predict_proba(X_b)[:, 1]
                for (s1_id, cand_id), p in zip(kept_b, probs):
                    if p >= threshold:
                        batch_matches[s1_id].append((cand_id, p))

        for s1_id, _, _, _ in batch:
            cand_list = batch_cands[s1_id]
            cand_set = set(cand_list)

            raw_matches = batch_matches.get(s1_id, [])
            ranked_matches = [cid for cid, _ in sorted(raw_matches, key=lambda t: t[1], reverse=True)]

            seen_m = set()
            final_matches = [m for m in ranked_matches if m in cand_set and not (m in seen_m or seen_m.add(m))]

            if final_matches:
                entities_with_matches += 1
                total_matches_predicted += len(final_matches)
            else:
                singletons_predicted += 1

            f_match.write(f"{s1_id}\t{','.join(final_matches)}\n")

        total_s1_processed += len(batch)
        del batch, batch_s1_lookup, batch_cands, pairs_to_score, batch_matches
        gc.collect()

    f_cand.close()
    f_match.close()

    total_time = time.time() - t0_start
    print("\n=== Inference Complete ===")
    print(f"Total S1 entities processed   : {total_s1_processed:,}")
    print(f"Total candidate pairs written  : {total_candidates_generated:,} (avg {total_candidates_generated/max(total_s1_processed,1):.2f}/S1)")
    print(f"Entities with predicted matches: {entities_with_matches:,}")
    print(f"Singletons (0 matches)         : {singletons_predicted:,}")
    print(f"Total matches predicted        : {total_matches_predicted:,}")
    print(f"Total inference runtime        : {total_time:.1f}s ({total_time/60:.2f} min)")
    print(f"Output files:\n  1. {cand_path}\n  2. {match_path}")


if __name__ == "__main__":
    main()
