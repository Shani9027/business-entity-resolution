"""
Text normalization utilities for business names and addresses.

Design goal: normalize away the *documented* noise patterns (legal-suffix
abbreviations, punctuation, common street-type abbreviations) using generic,
language-agnostic rules -- no country-specific hardcoding beyond a small,
easily-extensible abbreviation dictionary, and no external lookups of any
kind (per the challenge's fair-play rules).
"""
import re
import unicodedata

# Common legal-entity suffixes across locales. Matching on these lets us
# strip them for a "core name" comparison while still keeping the raw name
# for suffix-sensitive features.
# Common legal-entity suffixes across locales (US, India, UK, France, Germany, etc.).
# Matching on these lets us strip them for a "core name" comparison while still keeping
# the raw name for suffix-sensitive features.
LEGAL_SUFFIXES = [
    # Multi-word
    "private limited", "pvt ltd", "pvt. ltd.", "pvt", "ltd", "limited",
    "incorporated", "inc", "corporation", "corp", "co", "company",
    "llc", "llp", "l.l.c.", "plc", "gmbh",
    # French entity types (Phase 8 support)
    "sarl", "s.a.r.l.", "sas", "s.a.s.", "sasu", "s.a.s.u.",
    "sci", "s.c.i.", "eurl", "e.u.r.l.", "snc", "s.n.c.",
    "gie", "g.i.e.", "sa", "s.a.", "eirl", "scop",
]
# sort longest-first so multi-word suffixes are stripped before sub-tokens
LEGAL_SUFFIXES = sorted(set(LEGAL_SUFFIXES), key=len, reverse=True)

# PERF: precompile everything ONCE at import time. Recompiling/rebuilding a
# regex per suffix per record (as a naive implementation would) is the #1
# cause of multi-hour runtimes on multi-million-row datasets -- a single
# combined alternation pattern is 10-20x faster.
_SUFFIX_ALTERNATION = "|".join(
    re.escape(s).replace(r"\ ", r"\s+") for s in LEGAL_SUFFIXES
)
SUFFIX_PATTERN = re.compile(r"\b(?:" + _SUFFIX_ALTERNATION + r")\b")
PUNCT_PATTERN = re.compile(r"[^\w\s]")
WS_PATTERN = re.compile(r"\s+")
DIGIT_PATTERN = re.compile(r"\d+")

ADDRESS_ABBREV = {
    # English / US / India
    "road": "rd", "street": "st", "avenue": "ave", "boulevard": "blvd",
    "drive": "dr", "lane": "ln", "highway": "hwy", "apartment": "apt",
    "building": "bldg", "floor": "fl", "suite": "ste", "north": "n",
    "south": "s", "east": "e", "west": "w", "mount": "mt",
    "saint": "st", "square": "sq", "circle": "cir", "court": "ct",
    "place": "pl", "terrace": "ter", "parkway": "pkwy",
    # French address forms (Phase 8 support: R/Rue, AV/Avenue, BD/Boulevard, Allée, Chemin, Impasse, Cité)
    "rue": "rue", "r": "rue",
    "av": "ave",
    "bd": "blvd", "bvd": "blvd",
    "allee": "allee", "all": "allee",
    "chemin": "chemin", "che": "chemin", "ch": "chemin",
    "impasse": "impasse", "imp": "impasse",
    "cite": "cite",
    "route": "rte", "rte": "rte",
}

# Landmark regex patterns (Phase 2 fix):
# Safely remove landmark phrases without destroying the rest of the address when no comma exists.
_LANDMARK_TRIGGERS = r"(?:near|opp(?:osite)?|behind|next to|adjacent to)"
_LANDMARK_NOUNS = (
    r"(?:atm|station|hospital|school|college|bank|temple|masjid|hotel|mall|park|"
    r"bridge|garden|gate|market|tower|plaza|office|complex|petrol\s+pump|depot|"
    r"bus\s+stand|airport|metro|flyover|stop|stand|mandir|gurudwara|church)"
)
_ROAD_MARKERS = (
    r"(?:road|rd|street|st|ave|avenue|blvd|boulevard|lane|ln|drive|dr|way|hwy|"
    r"highway|marg|path|nagar|colony|sector|sec|cross|main|bldg|building|flat|"
    r"plot|block|floor|no|h\.?no|\d+)"
)

_LANDMARK_COMMA_RE = re.compile(rf"\b{_LANDMARK_TRIGGERS}\b\.?\s*[^,]+?,", re.IGNORECASE)
_LANDMARK_END_RE = re.compile(rf",\s*\b{_LANDMARK_TRIGGERS}\b\.?\s*[\w\s]{{1,40}}$", re.IGNORECASE)
_LANDMARK_NOUN_RE = re.compile(
    rf"\b{_LANDMARK_TRIGGERS}\b\.?\s+(?:(?:the|a)\s+)?(?:[\w\s]{{1,30}}?\s+)?{_LANDMARK_NOUNS}\b(?:\s+(?={_ROAD_MARKERS}\b))?",
    re.IGNORECASE,
)
_LANDMARK_ROAD_RE = re.compile(
    rf"\b{_LANDMARK_TRIGGERS}\b\.?\s+(?:(?:the|a)\s+)?(?:\w+\s+){{1,3}}(?={_ROAD_MARKERS}\b)",
    re.IGNORECASE,
)


def _strip_accents(text: str) -> str:
    """Fold accented / transliterated characters to plain ASCII where possible."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def basic_clean(text) -> str:
    if text is None or (isinstance(text, float)):
        return ""
    text = str(text)
    text = _strip_accents(text)
    # Collapse dotted initialisms (e.g. S.A.S. -> SAS, L.L.C. -> LLC, S.C.I. -> SCI)
    text = re.sub(r"\b([A-Za-z])\.(?=[A-Za-z]\b)", r"\1", text)
    text = text.lower()
    text = text.replace("&", " and ")
    text = PUNCT_PATTERN.sub(" ", text)  # drop punctuation
    text = WS_PATTERN.sub(" ", text).strip()
    return text


def clean_landmarks(text: str) -> str:
    """Safely remove landmark phrases without destroying useful address information."""
    s = _LANDMARK_COMMA_RE.sub(" ", text)
    s = _LANDMARK_END_RE.sub(" ", s)
    s = _LANDMARK_NOUN_RE.sub(" ", s)
    s = _LANDMARK_ROAD_RE.sub(" ", s)
    return s


def normalize_name(raw_name: str):
    """Returns (full_normalized_name, core_name_without_legal_suffix, token_set)."""
    cleaned = basic_clean(raw_name)
    core = SUFFIX_PATTERN.sub(" ", cleaned)
    core = WS_PATTERN.sub(" ", core).strip()
    tokens = frozenset(core.split()) if core else frozenset(cleaned.split())
    return cleaned, core, tokens


def normalize_address(raw_addr: str):
    """Returns (full_normalized_address, token_set, digit_tokens)."""
    if raw_addr is None or (isinstance(raw_addr, float)):
        raw_addr = ""
    text = str(raw_addr)
    text = _strip_accents(text)
    no_landmark = clean_landmarks(text)
    cleaned = basic_clean(no_landmark)
    words = cleaned.split()
    abbrev_words = [ADDRESS_ABBREV.get(w, w) for w in words]
    abbrev_text = " ".join(abbrev_words)
    tokens = frozenset(abbrev_words)
    # digits present in the address (house numbers / PIN / ZIP codes) --
    # useful as a strong, language-agnostic signal
    digit_tokens = frozenset(DIGIT_PATTERN.findall(text))
    return abbrev_text, tokens, digit_tokens


def name_ngrams(text: str, n: int = 3):
    text = text.replace(" ", "_")
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i + n] for i in range(len(text) - n + 1)}


# ---------------------------------------------------------------------------
# Phonetic key (Soundex-style, generic rule-based, no external data/lookups).
# Purpose: catch matches where a name/word is transliterated, misspelled, or
# spelled slightly differently but *sounds* the same (e.g. "Sharma" vs
# "Sharrma", or a transliteration variant) -- these break exact-token and
# even trigram overlap but usually still map to the same phonetic code.
# ---------------------------------------------------------------------------
_SOUNDEX_CODES = {
    "B": "1", "F": "1", "P": "1", "V": "1",
    "C": "2", "G": "2", "J": "2", "K": "2", "Q": "2", "S": "2", "X": "2", "Z": "2",
    "D": "3", "T": "3",
    "L": "4",
    "M": "5", "N": "5",
    "R": "6",
}


def soundex(word: str) -> str:
    """Classic Soundex: 1 letter + 3 digits. Empty/short input -> ''."""
    word = "".join(ch for ch in word.upper() if ch.isalpha())
    if not word:
        return ""
    first = word[0]
    codes = []
    prev_code = _SOUNDEX_CODES.get(first, "")
    for ch in word[1:]:
        code = _SOUNDEX_CODES.get(ch, "")
        if code and code != prev_code:
            codes.append(code)
        if ch not in ("H", "W"):
            prev_code = code
    result = (first + "".join(codes) + "000")[:4]
    return result


def name_phonetic_codes(tokens):
    """Soundex codes for name tokens of length >= 3 (short tokens are too
    ambiguous phonetically to be useful and would create huge posting lists)."""
    return frozenset(soundex(t) for t in tokens if len(t) >= 3 and soundex(t))


def name_prefixes(tokens, length: int = 4):
    """First-N-character prefix of each token (len >= length) -- catches
    plural/suffix-truncation variants that share a stem but not the whole word."""
    return frozenset(t[:length] for t in tokens if len(t) >= length)
