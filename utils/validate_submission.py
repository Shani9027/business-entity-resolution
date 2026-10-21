"""
validate_submission.py
-----------------------
Stdlib-only validator that checks matching_results.tsv and candidate_pairs.tsv
against the rules in the challenge problem statement:
  - exactly one row per Source-1 test entity
  - matched_entity_ids / candidate_entity_ids reference only S2-/S3- ids that
    exist in the test set
  - no duplicate ids within a single row's list
  - no duplicate source1_entity_id rows
  - matched ids are a subset of that row's candidate ids
  - no self-matches to Source 1

This is a convenience re-implementation for local iteration; always also run
the organizers' own utils/validate_submission.py before the final submission.

Usage:
    python3 validate_submission.py \
        --matching output/matching_results.tsv \
        --candidate output/candidate_pairs.tsv \
        --test-dir dataset/test
"""
import argparse
import csv
import os
import sys


def read_tsv(path):
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        rows = list(reader)
    if not rows:
        return [], []
    return rows[0], rows[1:]


def load_id_list_file(path):
    out = {}
    dupes = []
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, [])
        for r in reader:
            if not r:
                continue
            if len(r) == 1:
                r = r + [""]
            s1_id, ids_cell = r[0], r[1] if len(r) > 1 else ""
            if s1_id in out:
                dupes.append(s1_id)
            ids = [x for x in ids_cell.split(",") if x.strip()] if ids_cell else []
            out[s1_id] = ids
    return header, out, dupes


def load_valid_ids(test_dir):
    s1_ids, s2_ids, s3_ids = set(), set(), set()
    for fname, bucket in [("test_source1.tsv", s1_ids), ("test_source2.tsv", s2_ids),
                           ("test_source3.tsv", s3_ids)]:
        path = os.path.join(test_dir, fname)
        with open(path, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader, [])
            id_col = header.index("entity_id") if "entity_id" in header else 0
            for r in reader:
                if r:
                    bucket.add(r[id_col])
    return s1_ids, s2_ids, s3_ids


def validate(matching_path, candidate_path, test_dir):
    issues = []
    s1_ids, s2_ids, s3_ids = load_valid_ids(test_dir)
    valid_match_ids = s2_ids | s3_ids

    m_header, matches, m_dupes = load_id_list_file(matching_path)
    c_header, candidates, c_dupes = load_id_list_file(candidate_path)

    if m_header[:2] != ["source1_entity_id", "matched_entity_ids"]:
        issues.append(f"matching_results.tsv header must be "
                       f"'source1_entity_id\\tmatched_entity_ids', got {m_header}")
    if c_header[:2] != ["source1_entity_id", "candidate_entity_ids"]:
        issues.append(f"candidate_pairs.tsv header must be "
                       f"'source1_entity_id\\tcandidate_entity_ids', got {c_header}")

    for dup in m_dupes:
        issues.append(f"Duplicate source1_entity_id row in matching_results.tsv: {dup}")
    for dup in c_dupes:
        issues.append(f"Duplicate source1_entity_id row in candidate_pairs.tsv: {dup}")

    missing_in_matching = s1_ids - set(matches.keys())
    if missing_in_matching:
        issues.append(f"{len(missing_in_matching)} S1 test entities missing from "
                       f"matching_results.tsv, e.g. {list(missing_in_matching)[:5]}")

    extra_in_matching = set(matches.keys()) - s1_ids
    if extra_in_matching:
        issues.append(f"{len(extra_in_matching)} rows in matching_results.tsv reference "
                       f"S1 ids not in the test set, e.g. {list(extra_in_matching)[:5]}")

    for s1_id, ids in matches.items():
        if len(ids) != len(set(ids)):
            issues.append(f"Duplicate ids within matched_entity_ids for {s1_id}")
        bad = [i for i in ids if i not in valid_match_ids]
        if bad:
            issues.append(f"{s1_id}: matched_entity_ids contains invalid/self/S1 ids: {bad}")
        cand_set = set(candidates.get(s1_id, []))
        not_in_cand = [i for i in ids if i not in cand_set]
        if not_in_cand:
            issues.append(f"{s1_id}: matched ids not present in its candidate list "
                           f"(possible pipeline bug): {not_in_cand}")

    for s1_id, ids in candidates.items():
        if len(ids) != len(set(ids)):
            issues.append(f"Duplicate ids within candidate_entity_ids for {s1_id}")
        bad = [i for i in ids if i not in valid_match_ids]
        if bad:
            issues.append(f"{s1_id}: candidate_entity_ids contains invalid/self/S1 ids: {bad}")

    missing_in_candidates = s1_ids - set(candidates.keys())
    if missing_in_candidates:
        issues.append(f"{len(missing_in_candidates)} S1 test entities missing from "
                       f"candidate_pairs.tsv, e.g. {list(missing_in_candidates)[:5]}")

    return issues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matching", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--test-dir", required=True)
    args = ap.parse_args()

    issues = validate(args.matching, args.candidate, args.test_dir)
    if not issues:
        print("PASS")
        sys.exit(0)
    else:
        print(f"{len(issues)} issue(s) found:")
        for i, msg in enumerate(issues, 1):
            print(f"{i}. {msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()
