"""Record-level normalization: raw (name, address) -> normalized fields.

Fields produced per record (all strings, '' when absent):
  nm       normalized name (Latin ASCII, lower, punctuation-free, aliases applied)
  nm_a     first part of an alias construct ("X dba Y" -> X), else nm
  nm_b     second part ("X dba Y" -> Y), else ''
  nm_dom   core of a web domain / social handle used as the name ('' if none)
  nm_scr   script profile of the raw name (latin / brahmic / mixed / other / empty)
  ad       normalized address (Latin ASCII, lower, null tokens removed, aliases applied)
  ad_num   space-joined numeric tokens of the address (leading zeros stripped)
  ad_scr   script profile of the raw address

Maps are injected (module globals) so that multiprocessing workers inherit them via fork.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Optional, Sequence, Tuple

from .normalize import (
    SEED_TOKEN_MAP,
    _brahmic_base,
    basic_clean,
    script_profile,
    strip_punct,
)

# injected maps --------------------------------------------------------------
NATIVE_MAP: Dict[str, str] = {}        # native-script token -> latin token (learned)
NAME_ALIAS: Dict[str, str] = {}        # latin name token -> canonical (learned + seed)
ADDR_ALIAS: Dict[str, str] = {}        # latin address token -> canonical (learned)


def set_maps(native: Optional[Dict[str, str]] = None, name_alias: Optional[Dict[str, str]] = None,
             addr_alias: Optional[Dict[str, str]] = None) -> None:
    global NATIVE_MAP, NAME_ALIAS, ADDR_ALIAS
    NATIVE_MAP = dict(native or {})
    NAME_ALIAS = dict(name_alias or {})
    ADDR_ALIAS = dict(addr_alias or {})


_ALIAS_SPLIT = re.compile(
    r"(?i)\s+(?:dba|d/b/a|d\.b\.a\.?|aka|a/k/a|a\.k\.a\.?|fka|f/k/a|f\.k\.a\.?|"
    r"doing business as|trading as|t/a|formerly known as|also known as)\s+"
)
_HANDLE = re.compile(r"@([A-Za-z0-9_.À-ɏ-]{3,})")
_DOMAIN = re.compile(
    r"(?i)(?:https?://)?(?:www\.)?([a-z0-9À-ɏ][a-z0-9À-ɏ-]{1,62})"
    r"\.(?:com|net|org|co\.in|in|fr|co|biz|info|io|us|uk)\b"
)
_ZW = re.compile(r"[​-‍﻿]")
_LEET_MAP = {"0": "o", "1": "l", "3": "e", "5": "s", "4": "a", "7": "t"}
_ALPHA = re.compile(r"[a-z]")
_NUMS = re.compile(r"\d+")


def _has_brahmic(tok: str) -> bool:
    return any(_brahmic_base(ord(c)) is not None for c in tok)


def map_native_tokens(text: str) -> str:
    """Replace whole native-script tokens by their learned Latin form (if known)."""
    if not NATIVE_MAP or not text:
        return text
    out = []
    for raw in text.split():
        core = strip_punct(raw)
        if core and _has_brahmic(core):
            m = NATIVE_MAP.get(core)
            out.append(m if m else raw)
        else:
            out.append(raw)
    return " ".join(out)


def _deleet(tok: str) -> str:
    # only for name tokens that are mostly letters (keep unit codes like 135e, b46, 22nd)
    low = tok.lower()
    if any(c.isdigit() for c in low) and len(_ALPHA.findall(low)) >= 3:
        return "".join(_LEET_MAP.get(c, c) if c.isdigit() else c for c in tok)
    return tok


def _deleet_raw(text: str) -> str:
    """Leetspeak repair on raw name tokens before punctuation removal
    ('Téchn0logies' -> 'Téchnologies', 'W0ckhardt', 'High1and'); '@' only mid-word."""
    return " ".join(_deleet(w) for w in (text or "").split())


def _apply_alias(toks: List[str], amap: Dict[str, str]) -> List[str]:
    if not amap:
        return toks
    out: List[str] = []
    for w in toks:
        m = amap.get(w)
        if m is None:
            out.append(w)
        elif m:
            out.extend(m.split())
    return out


def _dedup_adjacent(toks: List[str]) -> List[str]:
    out: List[str] = []
    for w in toks:
        if not out or out[-1] != w or len(w) < 3:
            out.append(w)
    return out


def _name_tokens(text: str) -> List[str]:
    t = basic_clean(map_native_tokens(_deleet_raw(text)))
    toks = [w for w in t.split() if w]
    toks = _apply_alias(toks, NAME_ALIAS if NAME_ALIAS else SEED_TOKEN_MAP)
    return _dedup_adjacent([w for w in toks if w])


def norm_name_fields(raw: str) -> Tuple[str, str, str, str, str]:
    raw = _ZW.sub("", unicodedata.normalize("NFKC", raw or ""))
    scr = script_profile(raw)
    main = raw.split("|")[0]
    dom = ""
    m = _HANDLE.search(main)
    if m:
        dom = m.group(1)
    else:
        m = _DOMAIN.search(main)
        if m:
            dom = m.group(1)
    if dom:
        dom = re.sub(r"[^a-z0-9]", "", basic_clean(dom).replace(" ", ""))
    parts = [p for p in _ALIAS_SPLIT.split(main) if p.strip()]
    if not parts:
        parts = [main]
    if dom and (main.strip().startswith("@") or _DOMAIN.fullmatch(main.strip().lower() or "x") is not None
                or len(main.split()) == 1):
        nm_toks = [dom]
    else:
        nm_toks = _name_tokens(" ".join(parts))
    nm = " ".join(nm_toks)
    nm_a = " ".join(_name_tokens(parts[0])) if len(parts) > 1 else nm
    nm_b = " ".join(_name_tokens(" ".join(parts[1:]))) if len(parts) > 1 else ""
    return nm, nm_a, nm_b, dom, scr


def norm_addr_fields(raw: str) -> Tuple[str, str, str]:
    raw = _ZW.sub("", unicodedata.normalize("NFKC", raw or ""))
    scr = script_profile(raw)
    t = basic_clean(map_native_tokens(raw))
    toks = t.split()
    toks = _apply_alias(toks, ADDR_ALIAS)
    ad = " ".join(toks)
    nums = [n.lstrip("0") or "0" for n in _NUMS.findall(ad)]
    return ad, " ".join(nums), scr


def normalize_batch(batch: Sequence[Tuple[str, str]]) -> List[Tuple[str, ...]]:
    out = []
    for name, addr in batch:
        nm, nm_a, nm_b, dom, nscr = norm_name_fields(name)
        ad, adn, ascr = norm_addr_fields(addr)
        out.append((nm, nm_a, nm_b, dom, nscr, ad, adn, ascr))
    return out


FIELDS = ["nm", "nm_a", "nm_b", "nm_dom", "nm_scr", "ad", "ad_num", "ad_scr"]


def normalize_parallel(names: Sequence[str], addrs: Sequence[str], workers: int = 0,
                       chunk: int = 20000) -> Dict[str, List[str]]:
    """Normalize many records with a fork-based process pool (maps inherited)."""
    import multiprocessing as mp
    import os

    workers = workers or os.cpu_count() or 1
    pairs = list(zip(names, addrs))
    chunks = [pairs[i:i + chunk] for i in range(0, len(pairs), chunk)]
    if workers == 1 or len(chunks) == 1:
        res = [normalize_batch(c) for c in chunks]
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(workers) as pool:
            res = pool.map(normalize_batch, chunks, chunksize=1)
    cols: Dict[str, List[str]] = {f: [] for f in FIELDS}
    for r in res:
        for row in r:
            for f, v in zip(FIELDS, row):
                cols[f].append(v)
    return cols
