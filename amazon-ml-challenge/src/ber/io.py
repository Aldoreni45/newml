"""Reading challenge TSVs and writing submission TSVs.

All files are tab-separated with NO quoting. Fields are read as raw strings:
  - quote_char=None  -> a literal `"` inside a business name is data, not a quote
  - no NA inference  -> the literal strings "null", "NA", "None" stay as text
    (they occur inside addresses and must be handled by normalization, not the reader)
"""
from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Set

SOURCE_COLS = ["entity_id", "business_name", "business_address", "country"]
GT_COLS = ["source1_entity_id", "matched_entity_ids"]
MATCH_HEADER = ["source1_entity_id", "matched_entity_ids"]
CAND_HEADER = ["source1_entity_id", "candidate_entity_ids"]


def read_source(path: str):
    """Read one *_sourceN.tsv as a polars DataFrame of Utf8 columns ('' for empty)."""
    import polars as pl

    return pl.read_csv(
        path,
        separator="\t",
        quote_char=None,
        has_header=True,
        infer_schema=False,
        missing_utf8_is_empty_string=True,
        encoding="utf8",
    )


def read_ground_truth(path: str) -> Dict[str, Set[str]]:
    """Return {source1_id: set(matched S2/S3 ids)}; empty set for singletons.

    Stdlib parser so it works anywhere (the scorer's format is trivially line-based).
    """
    truth: Dict[str, Set[str]] = {}
    with open(path, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        assert [h.strip().lower() for h in header] == GT_COLS, header
        for line in f:
            s1, tab, rest = line.rstrip("\n").partition("\t")
            if not s1.strip():
                continue
            rest = rest.strip()
            truth[s1.strip()] = set(x.strip() for x in rest.split(",") if x.strip()) if rest else set()
    return truth


def read_id_list_file(path: str) -> Dict[str, Set[str]]:
    """Read a matching_results.tsv / candidate_pairs.tsv style file."""
    out: Dict[str, Set[str]] = {}
    with open(path, encoding="utf-8") as f:
        f.readline()
        for line in f:
            s1, tab, rest = line.rstrip("\n").partition("\t")
            if not s1.strip():
                continue
            rest = rest.strip()
            out[s1.strip()] = set(x for x in rest.split(",") if x) if rest else set()
    return out


def _order_ids(ids: Iterable[str], order: Optional[Dict[str, float]] = None) -> List[str]:
    ids = list(dict.fromkeys(ids))  # de-duplicate, keep first occurrence
    if order is not None:
        ids.sort(key=lambda x: -order.get(x, 0.0))
    return ids


def write_id_list_file(
    path: str,
    s1_ids: Iterable[str],
    mapping: Dict[str, Iterable[str]],
    kind: str = "matching",
    scores: Optional[Dict[str, Dict[str, float]]] = None,
) -> None:
    """Write one row per S1 id (in the given order), empty list when no ids.

    kind: "matching" -> header matched_entity_ids; "candidate" -> candidate_entity_ids.
    Guarantees: exactly one row per S1, no intra-list duplicates, only S2-/S3- ids.
    """
    header = MATCH_HEADER if kind == "matching" else CAND_HEADER
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    seen = set()
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(header) + "\n")
        for s1 in s1_ids:
            if s1 in seen:
                raise ValueError(f"duplicate S1 row {s1}")
            seen.add(s1)
            ids = [x for x in mapping.get(s1, ()) if x.startswith(("S2-", "S3-"))]
            ids = _order_ids(ids, scores.get(s1) if scores else None)
            f.write(f"{s1}\t{','.join(ids)}\n")
