"""Country-agnostic text normalization for business names and addresses.

Design principles (see SOLUTION_ARCHITECTURE.md):
  * Open-set countries: nothing here branches on a country label. Script handling is
    driven by Unicode properties of the text itself, so an unseen country/script still
    gets a sensible Latin-ASCII normal form.
  * Cross-script robustness: Indic scripts are transliterated to Latin with a
    deterministic rule-based scheme (inherent-vowel handling + virama), then everything
    goes through a phonetic "skeleton" that absorbs transliteration variance
    (aa/a, ee/i, sh/s, w/v, ph/f, doubled letters ...).
  * Abbreviation knowledge is NOT hard-coded per country. A tiny generic seed map is
    provided for the most universal business tokens; the rest is LEARNED from data
    (see ber.abbrev) and passed in as `token_map`.

Every function is pure-Python + stdlib (+ optional `anyascii`, ISC) so it can be applied
with polars `map_elements`/multiprocessing on the cloud box.
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Tuple

try:  # optional fallback for scripts other than Latin/Brahmic; anyascii is ISC-licensed
    from anyascii import anyascii as _unidecode
except Exception:  # pragma: no cover
    _unidecode = None

# --------------------------------------------------------------------------- Indic

# Unicode block starts for the Brahmic scripts that share the ISCII-derived layout:
# offsets inside each block encode the same phoneme, so one table covers all of them.
_BRAHMIC_BLOCKS = {
    0x0900: "devanagari", 0x0980: "bengali", 0x0A00: "gurmukhi", 0x0A80: "gujarati",
    0x0B00: "oriya", 0x0B80: "tamil", 0x0C00: "telugu", 0x0C80: "kannada", 0x0D00: "malayalam",
}
# offset -> latin (consonants carry an implicit 'a')
_VOWELS = {
    0x05: "a", 0x06: "a", 0x07: "i", 0x08: "i", 0x09: "u", 0x0A: "u", 0x0B: "ri", 0x0C: "li",
    0x0D: "e", 0x0E: "e", 0x0F: "e", 0x10: "ai", 0x11: "o", 0x12: "o", 0x13: "o", 0x14: "au",
    0x60: "rii", 0x61: "lii",
}
_MATRAS = {
    0x3E: "a", 0x3F: "i", 0x40: "i", 0x41: "u", 0x42: "u", 0x43: "ri", 0x44: "rii",
    0x45: "e", 0x46: "e", 0x47: "e", 0x48: "ai", 0x49: "o", 0x4A: "o", 0x4B: "o", 0x4C: "au",
    0x62: "li", 0x63: "lii", 0x57: "au",
}
_CONS = {
    0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "ng",
    0x1A: "ch", 0x1B: "chh", 0x1C: "j", 0x1D: "jh", 0x1E: "ny",
    0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh", 0x23: "n",
    0x24: "t", 0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "n",
    0x2A: "p", 0x2B: "ph", 0x2C: "b", 0x2D: "bh", 0x2E: "m",
    0x2F: "y", 0x30: "r", 0x31: "r", 0x32: "l", 0x33: "l", 0x34: "l", 0x35: "v",
    0x36: "sh", 0x37: "sh", 0x38: "s", 0x39: "h",
    0x58: "q", 0x59: "kh", 0x5A: "g", 0x5B: "z", 0x5C: "d", 0x5D: "rh", 0x5E: "f", 0x5F: "y",
}
_CHILLU = {0x7A: "n", 0x7B: "n", 0x7C: "r", 0x7D: "l", 0x7E: "l", 0x7F: "k", 0x54: "m", 0x55: "y", 0x56: "l"}
_VIRAMA = 0x4D
_ANUSVARA = 0x02
_CANDRABINDU = 0x01
_VISARGA = 0x03
_NUKTA = 0x3C
_DIGIT0 = 0x66


def _brahmic_base(cp: int) -> Optional[int]:
    base = cp & ~0x7F
    return base if base in _BRAHMIC_BLOCKS else None


def translit_indic(text: str) -> str:
    """Rule-based Brahmic -> Latin transliteration (ISO-15919-like, ASCII only).

    Handles inherent vowel, matras, virama (conjuncts), anusvara (-> n/m), nukta,
    native digits, and drops the word-final inherent 'a' (schwa deletion), which is
    how Indian names are conventionally romanized ("राम" -> "ram").
    Non-Brahmic characters pass through unchanged.
    """
    out: List[str] = []
    pending_a = False  # consonant emitted, inherent vowel not yet resolved

    def flush_a(word_end: bool) -> None:
        nonlocal pending_a
        if pending_a and not word_end:
            out.append("a")
        pending_a = False

    for ch in text:
        cp = ord(ch)
        base = _brahmic_base(cp)
        if base is None:
            flush_a(word_end=True)
            out.append(ch)
            continue
        off = cp - base
        if off in _CONS:
            flush_a(word_end=False)
            out.append(_CONS[off])
            pending_a = True
        elif off in _MATRAS:
            pending_a = False
            out.append(_MATRAS[off])
        elif off == _VIRAMA:
            pending_a = False
        elif off in _VOWELS:
            flush_a(word_end=False)
            out.append(_VOWELS[off])
        elif off in (_ANUSVARA, _CANDRABINDU):
            flush_a(word_end=False)
            out.append("n")
        elif off == _VISARGA:
            flush_a(word_end=False)
            out.append("h")
        elif off == _NUKTA:
            pass
        elif base == 0x0A00 and off == 0x70:      # Gurmukhi tippi = nasal
            flush_a(word_end=False)
            out.append("n")
        elif base == 0x0A00 and off == 0x71:      # Gurmukhi addak = gemination of next consonant
            pass
        elif base == 0x0D00 and off in _CHILLU:   # Malayalam chillu = dead consonant
            flush_a(word_end=False)
            out.append(_CHILLU[off])
        elif _DIGIT0 <= off <= _DIGIT0 + 9:
            flush_a(word_end=True)
            out.append(str(off - _DIGIT0))
        else:
            flush_a(word_end=True)
    flush_a(word_end=True)
    return "".join(out)


_NON_LATIN_RE = re.compile(r"[^\x00-\x7F]")
_SYMBOL_SPACE = {ord(c): " " for c in "°º№·•"}  # degree, ordinal, numero, bullets


def to_latin(text: str) -> str:
    """Any script -> Latin ASCII. Brahmic via our rules, Latin diacritics stripped,
    anything else via anyascii (if available) else dropped. Degree/numero signs -> space."""
    if not text or not _NON_LATIN_RE.search(text):
        return text
    t = translit_indic(text).translate(_SYMBOL_SPACE)
    t = unicodedata.normalize("NFKD", t)
    t = "".join(c for c in t if not unicodedata.combining(c))
    if _NON_LATIN_RE.search(t):
        if _unidecode is not None:
            t = _unidecode(t)
        else:
            t = _NON_LATIN_RE.sub(" ", t)
    return t


def script_profile(text: str) -> str:
    """Coarse script label of a string from Unicode data: 'latin', 'brahmic', 'mixed', 'other', 'empty'."""
    has_lat = has_bra = has_oth = False
    for ch in text:
        cp = ord(ch)
        if ch.isalpha():
            if cp < 0x250:
                has_lat = True
            elif _brahmic_base(cp) is not None:
                has_bra = True
            else:
                has_oth = True
    kinds = has_lat + has_bra + has_oth
    if kinds == 0:
        return "empty"
    if kinds > 1:
        return "mixed"
    return "latin" if has_lat else ("brahmic" if has_bra else "other")


# --------------------------------------------------------------------------- cleaning

_NULL_RE = re.compile(r"(?i)<?\b(?:null|none|nan|n/a|nil|undefined)\b>?")
_WS = re.compile(r"\s+")
_URL_PREFIX = re.compile(r"(?i)\b(?:https?://)?(?:www\.)")
_DOMAIN = re.compile(r"(?i)\b([a-z0-9][a-z0-9-]{1,62})\.(com|net|org|in|co\.in|fr|co|biz|info|io|us|uk)\b")
_APOS = re.compile(r"['’‘`´]")
_NONALNUM = re.compile(r"[^a-z0-9]+")
_SINGLE_RUN = re.compile(r"\b(?:[a-z] ){1,}[a-z]\b")


def strip_punct(tok: str) -> str:
    """Strip leading/trailing punctuation/symbols/separators by Unicode category.
    (Python's \w excludes Indic combining vowel signs, so regex \W-stripping would
    corrupt native-script tokens.)"""
    i, j = 0, len(tok)
    while i < j and unicodedata.category(tok[i])[0] in "PSZC":
        i += 1
    while j > i and unicodedata.category(tok[j - 1])[0] in "PSZC":
        j -= 1
    return tok[i:j]


def basic_clean(text: str) -> str:
    """NFKC -> Latin ASCII -> casefold -> null-tokens removed -> '&'->'and' ->
    apostrophes dropped -> every other non-alphanumeric char becomes a space ->
    runs of single letters collapsed ('l l c' -> 'llc', 'e e johnson' -> 'ee johnson').
    Applied identically to both sides of every comparison, so collapsing is consistent."""
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text)
    t = to_latin(t).casefold()
    t = _NULL_RE.sub(" ", t)
    t = _URL_PREFIX.sub(" ", t)
    t = t.replace("&", " and ")
    t = _APOS.sub("", t)
    t = _NONALNUM.sub(" ", t)
    t = _WS.sub(" ", t).strip()
    if t:
        t = _SINGLE_RUN.sub(lambda m: m.group(0).replace(" ", ""), t)
    return t


def tokens(text: str) -> List[str]:
    return text.split()


# --------------------------------------------------------------------------- phonetic

_PHON_RULES: Tuple[Tuple[str, str], ...] = (
    ("chh", ""), ("ch", ""), ("ck", "k"), ("c", "k"), ("", "c"),
    ("sh", "s"), ("ph", "f"), ("kh", "k"), ("gh", "g"),
    ("bh", "b"), ("dh", "d"), ("th", "t"), ("jh", "j"), ("q", "k"),
    ("w", "v"), ("z", "j"), ("x", "ks"), ("y", "i"), ("ee", "i"), ("oo", "u"),
    ("ou", "u"), ("aa", "a"), ("ii", "i"), ("uu", "u"),
)


@lru_cache(maxsize=1_000_000)
def phonetic(token: str) -> str:
    """Transliteration-invariant phonetic key for one ASCII token.

    Absorbs romanization variance: Shree/Shri/Sri, Aaditya/Aditya, Vikas/Wikas,
    Laxmi/Lakshmi (x->ks, sh->s => laksmi), Kh/K, doubled letters.
    """
    if not token or token.isdigit():
        return token
    t = token
    for a, b in _PHON_RULES:
        t = t.replace(a, b)
    t = re.sub(r"(.)\1+", r"\1", t)  # collapse repeats
    if len(t) > 3:
        head = "a" if t[0] in "aeiou" else t[0]  # any initial vowel -> 'a' (e/i/ee variance)
        t = head + re.sub(r"[aeiou]", "", t[1:])  # consonant skeleton
    return t


# --------------------------------------------------------------------------- abbreviations

# Minimal, universal seed; the full map is learned from data (ber.abbrev) and merged in.
SEED_TOKEN_MAP: Dict[str, str] = {
    "pvt": "private", "pvt.": "private", "ltd": "limited", "corp": "corporation",
    "co": "company", "inc": "incorporated", "intl": "international", "mfg": "manufacturing",
    "svcs": "services", "svc": "service", "bros": "brothers", "assoc": "associates",
    "mgmt": "management", "tech": "technology", "grp": "group",
}


def apply_token_map(toks: Iterable[str], token_map: Optional[Dict[str, str]]) -> List[str]:
    if not token_map:
        return list(toks)
    return [token_map.get(w, w) for w in toks]


# --------------------------------------------------------------------------- name / address

_DBA = re.compile(r"(?i)\s+(?:dba|d/b/a|doing business as|t/a|trading as|aka|a/k/a)\s+")


def split_dba(name: str) -> List[str]:
    """'Ectolumdrex dba X+ Madison Inc' -> ['Ectolumdrex', 'X+ Madison Inc']."""
    parts = [p for p in _DBA.split(name or "") if p.strip()]
    return parts if parts else [name or ""]


def name_domain_core(name: str) -> Optional[str]:
    """'heassociates.com' / 'x | www.shivshakti.com' -> 'heassociates' / 'shivshakti'."""
    m = _DOMAIN.search(name or "")
    return m.group(1).lower() if m else None


def norm_name(name: str, token_map: Optional[Dict[str, str]] = None) -> str:
    """Normalized name string: cleaned, Latin, abbreviations expanded, repeated tokens
    removed, pipe/web suffixes removed. Legal-form tokens are kept (removal is a
    separate, data-driven step: see `core_tokens`)."""
    raw = (name or "").split("|")[0]
    t = basic_clean(raw)
    toks = apply_token_map(tokens(t), token_map if token_map is not None else SEED_TOKEN_MAP)
    dedup: List[str] = []
    for w in toks:
        if not dedup or dedup[-1] != w:
            dedup.append(w)
    return " ".join(dedup)


def core_tokens(toks: List[str], generic: Optional[set]) -> List[str]:
    """Drop generic/legal tokens (learned from data as high-DF name tokens)."""
    if not generic:
        return toks
    core = [w for w in toks if w not in generic]
    return core if core else toks


def norm_address(addr: str, token_map: Optional[Dict[str, str]] = None) -> str:
    t = basic_clean(addr or "")
    toks = apply_token_map(tokens(t), token_map)
    return " ".join(toks)


_NUM = re.compile(r"\d+")


def numbers(text: str) -> List[str]:
    return _NUM.findall(text or "")


def postcode_candidates(addr: str) -> List[str]:
    """Pure digit-shape postcode detection, no country table: 5-6 digit runs and
    the '3+3' spaced form. Which shape is 'the postcode' in a country is learned from
    data statistics, not hard-coded."""
    a = addr or ""
    out = re.findall(r"(?<!\d)\d{5,6}(?!\d)", a)
    out += [x.replace(" ", "") for x in re.findall(r"(?<!\d)\d{3}\s\d{3}(?!\d)", a)]
    return out


def _selftest() -> None:
    assert translit_indic("राम") == "ram", translit_indic("राम")
    print("राम मार्केटिंग प्राइवेट लिमिटेड ->", norm_name("राम मार्केटिंग प्राइवेट लिमिटेड"))
    print("ஈஸ்டர்ன் கன்சல்டன்சி பிரைவேட் லிமிடெட் ->", norm_name("ஈஸ்டர்ன் கன்சல்டன்சி பிரைவேட் லிமிடெட்"))
    print("Sun पावर Provision ->", norm_name("Sun पावर Provision"))
    print("LLC Moncada Léarning Center ->", norm_name("LLC Moncada Léarning Center"))
    print("SHIVSHAKTI VIDYALAYA VIDYALAYA ... ->", norm_name("SHIVSHAKTI VIDYALAYA VIDYALAYA OVERSEAS CORPORATION | www.shivshakti.com"))
    print("addr ->", norm_address("G.t. Karnal Road, Industrial Area, New Delhi, null, A-68, दिल्ली"))
    print("phonetic:", [phonetic(w) for w in ["shree", "shri", "sri", "lakshmi", "laxmi", "aaditya", "aditya", "vikas", "wikas"]])
    print("postcodes:", postcode_candidates("Mumbai 400 001"), postcode_candidates("Tyler, TX 75701"))


if __name__ == "__main__":
    _selftest()
