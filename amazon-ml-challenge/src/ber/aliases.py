"""Learn normalization maps FROM THE PROVIDED DATA (no external dictionaries).

1. learn_native_map: native-script token -> Latin token, from train positive pairs
   where the target is written (partly) in an Indic script and the S1 record in Latin.
   Position alignment when token counts agree, Dice co-occurrence otherwise.
2. learn_abbrev_map: Latin abbreviation -> expansion (e.g. rd->road, tx->texas,
   pvt->private), mined from tokens that differ between the two sides of positive
   pairs, filtered by an abbreviation-shape test (same first letter, subsequence).
   Works for any country present in the pairs; for an unseen country the same miner can
   run on high-confidence pseudo-pairs (self-training) — no hard-coded country tables.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Tuple

from .normalize import _brahmic_base, strip_punct



def _is_native(tok: str) -> bool:
    return any(_brahmic_base(ord(c)) is not None for c in tok)


_ZW = re.compile("[​-‍﻿]")


def canonical_native(raw: str) -> str:
    """Same canonical form as records.map_native_tokens sees at lookup time (NFKC, no ZW chars)."""
    return _ZW.sub("", unicodedata.normalize("NFKC", raw or ""))


def _native_tokens(raw: str) -> List[str]:
    out = []
    for t in canonical_native(raw).split():
        c = strip_punct(t)
        if c:
            out.append(c)
    return out


def learn_native_map(
    pairs: Iterable[Tuple[str, str]],
    min_count: int = 2,
    min_share: float = 0.5,
    min_dice: float = 0.3,
) -> Tuple[Dict[str, str], Dict]:
    """pairs: (latin_normalized_s1_text, raw_target_text).

    Returns ({native_token: latin_token}, stats).
    """
    aligned: Dict[str, Counter] = defaultdict(Counter)
    co: Dict[str, Counter] = defaultdict(Counter)
    c_nat: Counter = Counter()
    c_lat: Counter = Counter()
    n_pairs = n_aligned = 0
    for lat, raw in pairs:
        ttoks = _native_tokens(raw)
        nat = [t for t in ttoks if _is_native(t)]
        if not nat:
            continue
        n_pairs += 1
        ltoks = lat.split()
        if len(ttoks) == len(ltoks):
            n_aligned += 1
            for a, b in zip(ttoks, ltoks):
                if _is_native(a):
                    aligned[a][b] += 1
        lset = set(ltoks)
        for a in set(nat):
            c_nat[a] += 1
            for b in lset:
                co[a][b] += 1
        for b in lset:
            c_lat[b] += 1
    out: Dict[str, str] = {}
    n_from_align = n_from_dice = 0
    for a in c_nat:
        cand = aligned.get(a)
        if cand:
            b, c = cand.most_common(1)[0]
            if c >= min_count and c / sum(cand.values()) >= min_share:
                out[a] = b
                n_from_align += 1
                continue
        if a in co:
            best, bd = None, 0.0
            for b, c in co[a].items():
                if c < min_count:
                    continue
                d = 2.0 * c / (c_nat[a] + c_lat[b])
                if d > bd:
                    best, bd = b, d
            if best is not None and bd >= min_dice:
                out[a] = best
                n_from_dice += 1
    stats = {"pairs_with_native": n_pairs, "aligned_pairs": n_aligned, "native_vocab": len(c_nat),
             "mapped": len(out), "from_alignment": n_from_align, "from_dice": n_from_dice}
    return out, stats


def is_abbrev(short: str, long: str) -> bool:
    """Abbreviation shape: same first letter, `short` is a subsequence of `long`, shorter."""
    if not short or not long or len(short) >= len(long) or short[0] != long[0]:
        return False
    if short.isdigit() or long.isdigit():
        return False
    it = iter(long)
    return all(ch in it for ch in short)


def learn_abbrev_map(
    pairs: Iterable[Tuple[str, str]],
    max_diff: int = 3,
    min_count: int = 20,
    min_dice: float = 0.2,
    min_prec: float = 0.3,
) -> Tuple[Dict[str, str], List[Tuple]]:
    """pairs: (s1_normalized_text, target_normalized_text) of TRUE matches.

    Counts (short, long) token pairs among the tokens that differ between the two sides,
    keeps pairs that look like abbreviations and co-occur consistently.
    Returns ({short: long}, ranked evidence list).
    """
    pc: Counter = Counter()
    c_short: Counter = Counter()
    c_long: Counter = Counter()
    for a, b in pairs:
        sa, sb = set(a.split()), set(b.split())
        ua, ub = sa - sb, sb - sa
        if not ua or not ub or len(ua) > max_diff or len(ub) > max_diff:
            continue
        for x in ua | ub:
            c_short[x] += 1
        for x in ua:
            for y in ub:
                if is_abbrev(y, x):
                    pc[(y, x)] += 1
                elif is_abbrev(x, y):
                    pc[(x, y)] += 1
    best: Dict[str, Tuple[str, int, float]] = {}
    evidence = []
    for (s, l), c in pc.items():
        if c < min_count:
            continue
        dice = 2.0 * c / (c_short[s] + c_short[l])
        prec = c / c_short[s]
        evidence.append((s, l, c, round(dice, 3), round(prec, 3)))
        if dice >= min_dice and prec >= min_prec:
            if s not in best or c > best[s][1]:
                best[s] = (l, c, dice)
    evidence.sort(key=lambda r: -r[2])
    return close_map({s: v[0] for s, v in best.items()}), evidence


def close_map(m: Dict[str, str], max_hops: int = 5) -> Dict[str, str]:
    """Resolve chains (a->b, b->c  =>  a->c) so one lookup pass is enough; drop cycles."""
    out = {}
    for a in m:
        b, hops, seen = m[a], 0, {a}
        while b in m and hops < max_hops and b not in seen:
            seen.add(b)
            b = m[b]
            hops += 1
        if b != a:
            out[a] = b
    return out
