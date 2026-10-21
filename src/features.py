"""
Feature engineering for candidate (Source1, Source2/3) pairs.

All features are computed purely from the two records' text fields -- no
external data, no geocoding, nothing looked up from the internet (per the
challenge's fair-play rules). Every feature is symmetric / generic so it
works the same way regardless of which country the pair belongs to,
including a country label never seen in training (e.g. France at test time).
"""
import re
import time
import numpy as np
from rapidfuzz import fuzz
from .normalize import normalize_name, normalize_address, name_ngrams

PROGRESS_EVERY = 500_000


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _digit_seq(text: str):
    return re.findall(r"\d+", str(text) if text is not None else "")


FEATURE_NAMES = [
    "name_token_jaccard",
    "name_trigram_jaccard",
    "name_levenshtein_ratio",
    "name_token_sort_ratio",
    "name_partial_ratio",
    "core_name_exact_match",
    "name_len_diff_ratio",
    "addr_token_jaccard",
    "addr_levenshtein_ratio",
    "addr_digit_jaccard",
    "addr_digit_exact_overlap_count",
    "addr_partial_ratio",
    "country_exact_match",
    "combined_min",
    "combined_max",
    "combined_avg",
]


def compute_pair_features(name1, addr1, country1, name2, addr2, country2) -> dict:
    _, core1, tokens1 = normalize_name(name1)
    _, core2, tokens2 = normalize_name(name2)
    tri1, tri2 = name_ngrams(core1), name_ngrams(core2)

    addr_norm1, addr_tokens1, addr_digits1 = normalize_address(addr1)
    addr_norm2, addr_tokens2, addr_digits2 = normalize_address(addr2)

    name_token_jaccard = _jaccard(tokens1, tokens2)
    name_trigram_jaccard = _jaccard(tri1, tri2)
    name_lev = fuzz.ratio(core1, core2) / 100.0
    name_tsort = fuzz.token_sort_ratio(core1, core2) / 100.0
    name_partial = fuzz.partial_ratio(core1, core2) / 100.0
    core_exact = 1.0 if core1 and core1 == core2 else 0.0
    len1, len2 = len(core1), len(core2)
    name_len_diff_ratio = 1.0 - (abs(len1 - len2) / max(len1, len2, 1))

    addr_token_jaccard = _jaccard(addr_tokens1, addr_tokens2)
    addr_lev = fuzz.ratio(addr_norm1, addr_norm2) / 100.0
    addr_digit_jaccard = _jaccard(addr_digits1, addr_digits2)
    addr_digit_overlap = len(addr_digits1 & addr_digits2)
    addr_partial = fuzz.partial_ratio(addr_norm1, addr_norm2) / 100.0

    country_match = 1.0 if str(country1).strip().lower() == str(country2).strip().lower() else 0.0

    core_sims = [name_lev, name_tsort, addr_lev]
    combined_min = float(np.min(core_sims))
    combined_max = float(np.max(core_sims))
    combined_avg = float(np.mean(core_sims))

    return {
        "name_token_jaccard": name_token_jaccard,
        "name_trigram_jaccard": name_trigram_jaccard,
        "name_levenshtein_ratio": name_lev,
        "name_token_sort_ratio": name_tsort,
        "name_partial_ratio": name_partial,
        "core_name_exact_match": core_exact,
        "name_len_diff_ratio": name_len_diff_ratio,
        "addr_token_jaccard": addr_token_jaccard,
        "addr_levenshtein_ratio": addr_lev,
        "addr_digit_jaccard": addr_digit_jaccard,
        "addr_digit_exact_overlap_count": float(addr_digit_overlap),
        "addr_partial_ratio": addr_partial,
        "country_exact_match": country_match,
        "combined_min": combined_min,
        "combined_max": combined_max,
        "combined_avg": combined_avg,
    }


def build_feature_matrix(pairs, record_lookup, s1_lookup=None):
    """
    pairs: list of (s1_id, cand_id) or (s1_id, cand_id, label)
    record_lookup: dict id -> (name, addr[, country])
    s1_lookup: optional separate dict for s1_id -> (name, addr[, country])
    Returns:
        If pairs have labels: (X, kept_pairs, y)
        If pairs do not have labels: (X, kept_pairs)
    """
    rows = []
    kept_pairs = []
    kept_labels = []
    has_labels = len(pairs) > 0 and len(pairs[0]) >= 3
    t0 = time.time()

    for i, item in enumerate(pairs, 1):
        if has_labels:
            s1_id, cand_id, label = item[0], item[1], item[2]
        else:
            s1_id, cand_id = item[0], item[1]
            label = None

        s1_info = s1_lookup.get(s1_id) if s1_lookup is not None else record_lookup.get(s1_id)
        cand_info = record_lookup.get(cand_id)

        if s1_info is None or cand_info is None:
            continue

        n1, a1 = s1_info[0], s1_info[1]
        c1 = s1_info[2] if len(s1_info) > 2 else "unknown"

        n2, a2 = cand_info[0], cand_info[1]
        c2 = cand_info[2] if len(cand_info) > 2 else c1

        feats = compute_pair_features(n1, a1, c1, n2, a2, c2)
        rows.append([feats[f] for f in FEATURE_NAMES])
        kept_pairs.append((s1_id, cand_id))
        if has_labels:
            kept_labels.append(label)

        if i % PROGRESS_EVERY == 0:
            elapsed = time.time() - t0
            print(f"  [features] processed {i:,}/{len(pairs):,} pairs "
                  f"({i/elapsed:,.0f}/sec)", flush=True)

    X = np.array(rows, dtype=np.float32) if rows else np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
    assert len(X) == len(kept_pairs), f"Feature matrix length {len(X)} != kept pairs length {len(kept_pairs)}"

    if has_labels:
        y = np.array(kept_labels, dtype=np.int32)
        assert len(X) == len(y), f"Feature matrix length {len(X)} != label length {len(y)}"
        return X, kept_pairs, y

    return X, kept_pairs
