"""
Block selection helpers and pretty-print utilities.
"""

from __future__ import annotations
from typing import List, Dict, Iterable
import re
import pandas as pd


def pretty_print_blocks(blocks: List[Dict[str, object]]) -> None:
    """
    Print a readable summary of detected blocks (as produced by your block detector).
    """
    for b in blocks:
        step = b["start"]
        if b["count"] > 1:
            print(f"Step {step}: {b['count']} x ({', '.join(b['block'])})")
        else:
            print(f"Step {step}: {', '.join(b['block'])}")


def block_ids_from_doe(doe_df: pd.DataFrame) -> List[int]:
    """Return sorted unique block_id values present in DoE."""
    if "block_id" not in doe_df.columns:
        return []
    return sorted([int(x) for x in pd.unique(doe_df["block_id"].dropna())])


def build_signature_map(blocks: List[Dict[str, object]]) -> Dict[int, str]:
    """
    Create a map from block_id (inferred by enumeration order) to signature string.

    Note
    ----
    If your DoE assigns block_id independently, prefer providing an explicit map.
    This fallback maps 1..N to the printed 'block' signatures in order of appearance.
    """
    mapping = {}
    bid = 1
    for b in blocks:
        signature = ", ".join(b["block"]) if isinstance(b.get("block"), list) else str(b.get("block"))
        mapping[bid] = signature
        bid += 1
    return mapping


def filter_block_ids_by_signature_regex(
    signature_map: Dict[int, str], pattern: str
) -> List[int]:
    """Return block_ids whose signatures match the given regex (case-insensitive)."""
    rx = re.compile(pattern, re.IGNORECASE)
    return [bid for bid, sig in signature_map.items() if rx.search(sig or "")]
