"""
KPI computation orchestration for selected blocks.
"""

from __future__ import annotations
from typing import List, Dict, Optional
import pandas as pd

from batterydata.pipeline.kpi_engine import (
    compute_multi_cycle_kpi_table, add_capacity_retention, add_energy_retention
)


def _select_cycle_frames(seg: pd.DataFrame, doe: pd.DataFrame, block_id: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Select raw/seg rows for a given block_id using DoE.step_number mapping.
    """
    doe_cycle = doe[doe["block_id"] == block_id]
    if doe_cycle.empty:
        return pd.DataFrame(), pd.DataFrame()
    mask = seg["step_number"].isin(doe_cycle["step_number"].tolist())
    raw_cycle = seg[mask]
    return raw_cycle, doe_cycle


def run_kpis_for_blocks(
    cell_name: str,
    seg_df: pd.DataFrame,
    doe_df: pd.DataFrame,
    block_ids: List[int],
    kpis: List[object],
    add_retentions: bool = True,
) -> pd.DataFrame:
    """
    Compute KPIs for each selected block_id, concatenate, and annotate with cell/block info.

    Parameters
    ----------
    cell_name : str
        Cell identifier (used to annotate output).
    seg_df : DataFrame
        Segmented raw table.
    doe_df : DataFrame
        DoE table containing 'block_id' and 'step_number'.
    block_ids : list[int]
        Which block IDs to compute.
    kpis : list[object]
        Instantiated KPI objects.
    add_retentions : bool
        If True, add capacity/energy retention columns.

    Returns
    -------
    DataFrame
        Concatenated KPI rows across all requested blocks.
    """
    parts: List[pd.DataFrame] = []
    for bid in block_ids:
        raw_cycle, doe_cycle = _select_cycle_frames(seg_df, doe_df, bid)
        if raw_cycle.empty or doe_cycle.empty:
            continue

        tbl = compute_multi_cycle_kpi_table(raw_df=raw_cycle, doe_df=doe_cycle, kpi_units=kpis)

        if add_retentions:
            if "Q_discharge_Ah" in tbl.columns:
                tbl = add_capacity_retention(tbl, discharge_col="Q_discharge_Ah", block_col="block_id")
            if "E_discharge_Wh" in tbl.columns:
                tbl = add_energy_retention(tbl, energy_col="E_discharge_Wh", block_col="block_id")

        # annotate
        tbl.insert(0, "cell_name", cell_name)
        tbl.insert(1, "block_id", bid)

        parts.append(tbl)

    if not parts:
        return pd.DataFrame()

    return pd.concat(parts, ignore_index=True)


def simple_sanity_checks(df: pd.DataFrame) -> Dict[str, str]:
    """
    Lightweight sanity flags (optional).
    Returns a dict of warnings (empty if none).
    """
    warnings: Dict[str, str] = {}

    if "CoulombicEfficiency" in df.columns:
        ce = df["CoulombicEfficiency"].dropna()
        if not ce.empty and ((ce < 0).any() or (ce > 1.2).any()):
            warnings["CE_range"] = "CoulombicEfficiency out of [0, 1.2] detected."

    if "EnergyEfficiency" in df.columns:
        ee = df["EnergyEfficiency"].dropna()
        if not ee.empty and ((ee < 0).any() or (ee > 1.2).any()):
            warnings["EE_range"] = "EnergyEfficiency out of [0, 1.2] detected."

    if "cccv_vs_total_mismatch_%" in df.columns:
        mm = df["cccv_vs_total_mismatch_%"].dropna()
        if not mm.empty and (mm.abs() > 5).any():
            warnings["CCCV_mismatch"] = "CC/CV vs total mismatch > 5% for some cycles."

    return warnings
