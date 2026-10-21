"""
Candidate generation (blocking) stage -- optimized for multi-million-row
datasets.

Changes vs. the earlier version (which was fine for small/demo data but too
slow/memory-heavy at ~10M+ total records):
  - Flat dict-of-list indices instead of nested defaultdict(defaultdict(set));
    far less per-key Python object overhead.
  - The character-trigram inverted index (which stores O(len(name)) postings
    PER record and gets huge/slow at this scale) is replaced by a cheap
    "sorted neighborhood" pass: core names are sorted per country, and a
    small window around each Source-1 name's insertion point is taken as a
    typo-tolerant fallback. This is O(n log n) instead of effectively
    O(n * avg_trigrams_per_name) index entries.
  - Progress is printed every PROGRESS_EVERY records while indexing, so a
    long run is visibly making progress instead of looking frozen.

Strategy recap:
  1. Group by the raw `country` string (open-ended -- no hardcoded list).
  2. Within each country: token index (exact word overlap) + digit index
     (PIN/ZIP/house-number exact overlap) + sorted-neighborhood window
     (typo tolerance).
  3. Union + weight the three signals per Source-1 entity, cap at
     MAX_CANDIDATES_PER_ENTITY.
"""
import bisect
import time
from collections import defaultdict, Counter
from .normalize import normalize_name, normalize_address, name_phonetic_codes, name_prefixes

MAX_CANDIDATES_PER_ENTITY = 60
MAX_POSTING_LIST_FRACTION = 0.05
MAX_POSTING_LIST_ABSOLUTE = 1000
NEIGHBORHOOD_WINDOW = 8       # neighbors on each side in the sorted-name pass (fallback only)
MIN_STRONG_CANDIDATES = 8     # only use neighborhood fallback if strong rules produce fewer than this
PROGRESS_EVERY = 500_000

ADDRESS_STOPWORDS = frozenset([
    "rd", "road", "st", "street", "ave", "avenue", "blvd", "boulevard",
    "ln", "lane", "dr", "drive", "way", "hwy", "highway", "ct", "court",
    "pl", "place", "sq", "square", "cir", "circle", "ter", "terrace",
    "pkwy", "parkway", "apt", "apartment", "ste", "suite", "fl", "floor",
    "bldg", "building", "no", "number", "hno", "plot", "flat", "block",
    "sector", "sec", "phase", "ph", "cross", "main", "opp", "near", "behind",
    "dist", "district", "city", "state", "post", "po", "at", "in", "of", "the",
    "and", "to", "rue", "r", "av", "bd", "bvd", "allee", "all", "chemin",
    "che", "ch", "impasse", "imp", "cite", "de", "la", "le", "du", "des",
    "d", "l", "sur", "sous", "les", "et", "aux", "en",
    "north", "south", "east", "west", "n", "s", "e", "w"
])


class BlockingIndex:
    def __init__(self, max_candidates=MAX_CANDIDATES_PER_ENTITY):
        # country -> name_token -> list(ids)
        self.token_index = defaultdict(lambda: defaultdict(list))
        # country -> addr_token -> list(ids)
        self.addr_token_index = defaultdict(lambda: defaultdict(list))
        # country -> digit (len >= 3) -> list(ids)
        self.digit_index = defaultdict(lambda: defaultdict(list))
        # country -> short_digit_addr_token -> list(ids)
        self.compound_digit_index = defaultdict(lambda: defaultdict(list))
        # country -> soundex(name_token) -> list(ids)  (typo/transliteration tolerance)
        self.phonetic_index = defaultdict(lambda: defaultdict(list))
        # country -> name_token[:4] -> list(ids)  (stem/suffix-truncation tolerance)
        self.prefix_index = defaultdict(lambda: defaultdict(list))
        # country -> list of (core_name, id) for fallback neighborhood pass
        self._sorted_pool = defaultdict(list)
        self.country_sizes = Counter()
        self.max_candidates = max_candidates
        self._finalized = False

    def add_records(self, df_or_records, id_col="entity_id", name_col="business_name",
                     addr_col="business_address", country_col="country",
                     record_lookup=None):
        """
        df_or_records can be a pandas DataFrame or an iterable of (rid, name, addr, country) tuples.
        If record_lookup dict is provided, populates record_lookup[rid] = (name, addr) simultaneously.
        """
        t0 = time.time()
        n = 0

        if hasattr(df_or_records, "itertuples"):
            cols = [id_col, name_col, addr_col, country_col]
            iterator = df_or_records[cols].itertuples(index=False, name=None)
        else:
            iterator = df_or_records

        for rid, name, addr, country in iterator:
            country = str(country)
            if record_lookup is not None:
                record_lookup[rid] = (name, addr)

            _, core, tokens = normalize_name(name)
            _, addr_tokens, digit_tokens = normalize_address(addr)

            self.country_sizes[country] += 1

            # 1. Name tokens (core name tokens, len >= 2)
            tok_idx = self.token_index[country]
            for t in tokens:
                if len(t) >= 2:
                    tok_idx[t].append(rid)

            # 2. Informative address tokens (non-stopwords, len >= 3)
            clean_addr_toks = [w for w in addr_tokens if len(w) >= 3 and w not in ADDRESS_STOPWORDS]
            a_idx = self.addr_token_index[country]
            for w in clean_addr_toks:
                a_idx[w].append(rid)

            # 3. Digits (len >= 3) & 4. Compound short digits (len 1-2) with address tokens
            d_idx = self.digit_index[country]
            short_digits = []
            for d in digit_tokens:
                if len(d) >= 3:
                    d_idx[d].append(rid)
                elif len(d) in (1, 2):
                    short_digits.append(d)

            if short_digits and clean_addr_toks:
                c_idx = self.compound_digit_index[country]
                for d in short_digits:
                    for w in clean_addr_toks[:2]:  # cap to top 2 informative tokens to keep posting compact
                        c_idx[f"{d}_{w}"].append(rid)

            # 5. Phonetic codes + prefixes of name tokens (typo/transliteration tolerance)
            ph_idx = self.phonetic_index[country]
            for code in name_phonetic_codes(tokens):
                ph_idx[code].append(rid)
            pre_idx = self.prefix_index[country]
            for pfx in name_prefixes(tokens):
                pre_idx[pfx].append(rid)

            self._sorted_pool[country].append((core, rid))

            n += 1
            if n % PROGRESS_EVERY == 0:
                elapsed = time.time() - t0
                print(f"  [blocking] indexed {n:,} records "
                      f"({n/elapsed:,.0f} records/sec)", flush=True)

        print(f"  [blocking] finished indexing {n:,} records in {time.time()-t0:.1f}s", flush=True)

    def _prune_common_postings(self):
        for idx in (self.token_index, self.addr_token_index, self.digit_index,
                    self.compound_digit_index, self.phonetic_index, self.prefix_index):
            for country, posting in idx.items():
                pct_limit = int(self.country_sizes[country] * MAX_POSTING_LIST_FRACTION)
                limit = max(5, min(pct_limit, MAX_POSTING_LIST_ABSOLUTE))
                for key in list(posting.keys()):
                    if len(posting[key]) > limit:
                        del posting[key]

    def finalize(self):
        if self._finalized:
            return
        print("  [blocking] pruning overly common tokens/digits ...", flush=True)
        self._prune_common_postings()
        print("  [blocking] sorting name pool per country for neighborhood lookup ...", flush=True)
        for country, pool in self._sorted_pool.items():
            pool.sort(key=lambda t: t[0])
        self._finalized = True
        print("  [blocking] index ready.", flush=True)

    def _neighborhood_candidates(self, country, core_name):
        pool = self._sorted_pool.get(country)
        if not pool:
            return []
        pos = bisect.bisect_left(pool, core_name, key=lambda t: t[0])
        lo = max(0, pos - NEIGHBORHOOD_WINDOW)
        hi = min(len(pool), pos + NEIGHBORHOOD_WINDOW)
        return [pool[i][1] for i in range(lo, hi)]

    def candidates_for(self, name, addr, country):
        country = str(country)
        _, core, tokens = normalize_name(name)
        _, addr_tokens, digit_tokens = normalize_address(addr)

        scored = Counter()

        # 1. Name token matches (+4)
        tok_idx = self.token_index.get(country, {})
        for t in tokens:
            if len(t) >= 2:
                for rid in tok_idx.get(t, ()):
                    scored[rid] += 4

        # 2. Informative address token matches (+2)
        clean_addr_toks = [w for w in addr_tokens if len(w) >= 3 and w not in ADDRESS_STOPWORDS]
        a_idx = self.addr_token_index.get(country, {})
        for w in clean_addr_toks:
            for rid in a_idx.get(w, ()):
                scored[rid] += 2

        # 3. Digits len >= 3 (+3)
        dig_idx = self.digit_index.get(country, {})
        short_digits = []
        for d in digit_tokens:
            if len(d) >= 3:
                for rid in dig_idx.get(d, ()):
                    scored[rid] += 3
            elif len(d) in (1, 2):
                short_digits.append(d)

        # 4. Compound short digits + address token (+4)
        if short_digits and clean_addr_toks:
            c_idx = self.compound_digit_index.get(country, {})
            for d in short_digits:
                for w in clean_addr_toks[:2]:
                    for rid in c_idx.get(f"{d}_{w}", ()):
                        scored[rid] += 4

        # 5. Phonetic (soundex) match on name tokens (+2) -- catches
        # transliteration/typo variants that break exact token overlap.
        ph_idx = self.phonetic_index.get(country, {})
        for code in name_phonetic_codes(tokens):
            for rid in ph_idx.get(code, ()):
                scored[rid] += 2

        # 6. Name-token prefix match (+1) -- catches stem/suffix-truncation
        # variants (e.g. plural forms, shortened words).
        pre_idx = self.prefix_index.get(country, {})
        for pfx in name_prefixes(tokens):
            for rid in pre_idx.get(pfx, ()):
                scored[rid] += 1

        # 7. Conditional neighborhood fallback:
        # Only inject sorted-neighborhood candidates when strong blocking signals found too few candidates
        if len(scored) < MIN_STRONG_CANDIDATES:
            for rid in self._neighborhood_candidates(country, core):
                scored[rid] += 1

        if not scored:
            return []

        ranked = sorted(scored.items(), key=lambda kv: kv[1], reverse=True)
        return [rid for rid, _ in ranked[:self.max_candidates]]


def build_blocking_index(*dfs, id_col="entity_id", name_col="business_name",
                          addr_col="business_address", country_col="country",
                          record_lookup=None, max_candidates=MAX_CANDIDATES_PER_ENTITY):
    index = BlockingIndex(max_candidates=max_candidates)
    for df in dfs:
        index.add_records(df, id_col=id_col, name_col=name_col,
                           addr_col=addr_col, country_col=country_col,
                           record_lookup=record_lookup)
    index.finalize()
    return index


def generate_candidate_pairs(source1_df, index: BlockingIndex,
                              id_col="entity_id", name_col="business_name",
                              addr_col="business_address", country_col="country"):
    """Returns dict: s1_entity_id -> list[candidate_entity_id]."""
    result = {}
    cols = [id_col, name_col, addr_col, country_col]
    sub = source1_df[cols]
    n = 0
    t0 = time.time()
    for s1_id, name, addr, country in sub.itertuples(index=False, name=None):
        result[s1_id] = index.candidates_for(name, addr, country)
        n += 1
        if n % PROGRESS_EVERY == 0:
            elapsed = time.time() - t0
            print(f"  [blocking] generated candidates for {n:,} S1 entities "
                  f"({n/elapsed:,.0f}/sec)", flush=True)
    return result
