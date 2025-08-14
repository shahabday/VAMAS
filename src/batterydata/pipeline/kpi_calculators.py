
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Protocol, Optional
import numpy as np
import pandas as pd


# ======================================================================
# Single-cycle engine (no ontology): integrates Q/E, computes KPIs you request
# ======================================================================

class SingleCycleKPI(Protocol):
    """A KPI unit that uses the single-cycle context and returns a dict."""
    name: str
    requires_basis: Tuple[str, ...]  # entries expected in ctx.cache

    def compute(self, ctx: "CycleContext") -> Dict[str, Any]:
        ...


@dataclass
class CycleContext:
    """Holds everything a KPI unit may need for one cycle."""
    raw: pd.DataFrame  # rows for exactly one cycle (charge/OCV/discharge)
    doe: pd.DataFrame  # DOE rows for the same cycle
    ids: Dict[str, Any] = field(default_factory=dict)    # block_id, cycle_index
    cache: Dict[str, Any] = field(default_factory=dict)  # basis features & flags

    def get_id_tuple(self) -> Tuple[Any, Any]:
        return self.ids.get("block_id"), self.ids.get("cycle_index")


def _integrate_trapz_abs(df: pd.DataFrame) -> Tuple[float, float, int, int]:
    """
    Integrate |I| and |I*V| over strictly positive dt (trapezoids).
    Expects: time_s, current_a, voltage_v
    Returns: (Q_Ah, E_Wh, n_rows_used, n_pairs_used)
    """
    needed = {"time_s", "current_a", "voltage_v"}
    if not needed.issubset(df.columns):
        raise KeyError(f"Missing columns for integration: {needed - set(df.columns)}")

    w = df[["time_s", "current_a", "voltage_v"]].dropna().sort_values("time_s")
    if len(w) < 2:
        return 0.0, 0.0, int(len(w)), 0

    t = w["time_s"].to_numpy()
    I = np.abs(w["current_a"].to_numpy())
    V = w["voltage_v"].to_numpy()

    dt = np.diff(t)
    mask = dt > 0
    if not np.any(mask):
        return 0.0, 0.0, int(len(w)), 0

    # Midpoint values for trapezoids
    I_mid = 0.5 * (I[:-1] + I[1:])
    IV_mid = 0.5 * (np.abs(I[:-1] * V[:-1]) + np.abs(I[1:] * V[1:]))

    Q_As = np.sum(I_mid[mask] * dt[mask])   # A*s
    E_Ws = np.sum(IV_mid[mask] * dt[mask])  # W*s

    return float(Q_As / 3600.0), float(E_Ws / 3600.0), int(len(w)), int(np.sum(mask))


def _sum_over_contiguous_runs(df: pd.DataFrame, step_kind: str) -> Tuple[float, float, Dict[str, int]]:
    """
    Sum Q/E over contiguous runs where step_type == step_kind (case-insensitive).
    """
    if df.empty:
        return 0.0, 0.0, {"runs": 0, "rows": 0, "pairs": 0}

    s = df.copy()
    s["_step_lc"] = s["step_type"].astype(str).str.lower()
    s["_run"] = (s["_step_lc"] != s["_step_lc"].shift(1)).cumsum()

    total_Q = total_E = 0.0
    runs = rows = pairs = 0
    target = step_kind.lower()

    for _, sub in s.groupby("_run", sort=True):
        if sub["_step_lc"].iloc[0] != target:
            continue
        Q, E, n_rows, n_pairs = _integrate_trapz_abs(sub)
        total_Q += Q
        total_E += E
        runs += 1
        rows += n_rows
        pairs += n_pairs

    s.drop(columns=["_step_lc", "_run"], inplace=True, errors="ignore")
    return total_Q, total_E, {"runs": runs, "rows": rows, "pairs": pairs}


def _sum_measured_step_ah_over_runs(df: pd.DataFrame, step_kind: str) -> Tuple[float, Dict[str, int]]:
    """
    Sum device-integrated step charge (Ah) over contiguous runs where step_type == step_kind,
    using 'step_charge_ah'. Robust to non-zero start by using abs(last - first).
    Returns (Q_meas_Ah, stats). If column missing, returns (NaN, {...}).
    """
    if "step_charge_ah" not in df.columns:
        return float("nan"), {"runs": 0, "rows": 0}

    if df.empty:
        return 0.0, {"runs": 0, "rows": 0}

    s = df.copy()
    s["_step_lc"] = s["step_type"].astype(str).str.lower()
    s["_run"] = (s["_step_lc"] != s["_step_lc"].shift(1)).cumsum()

    total_Q = 0.0
    runs = rows = 0
    target = step_kind.lower()

    for _, sub in s.groupby("_run", sort=True):
        if sub["_step_lc"].iloc[0] != target:
            continue
        sub2 = sub[["time_s", "step_charge_ah"]].dropna().sort_values("time_s")
        if len(sub2) == 0:
            continue
        q_first = float(sub2["step_charge_ah"].iloc[0])
        q_last  = float(sub2["step_charge_ah"].iloc[-1])
        total_Q += abs(q_last - q_first)
        runs += 1
        rows += len(sub2)

    s.drop(columns=["_step_lc", "_run"], inplace=True, errors="ignore")
    return float(total_Q), {"runs": runs, "rows": rows}


def _extract_ids(ctx: CycleContext) -> None:
    """
    Fill ctx.ids with cycle_index and block_id, preferring DOE; fallback to RAW.
    Accepts 'cycle_index' or 'cycle_number' for the cycle id. 'block_id' optional.
    """
    cyc_col: Optional[str] = None
    src = None
    for name in ("cycle_index", "cycle_number"):
        if name in ctx.doe.columns:
            cyc_col, src = name, ctx.doe
            break
        if name in ctx.raw.columns:
            cyc_col, src = name, ctx.raw
            break

    blk_col = None
    blk_src = None
    if "block_id" in ctx.doe.columns:
        blk_col, blk_src = "block_id", ctx.doe
    elif "block_id" in ctx.raw.columns:
        blk_col, blk_src = "block_id", ctx.raw

    cyc_val = None
    if cyc_col:
        u = pd.unique(src[cyc_col].dropna())
        if len(u) >= 1:
            cyc_val = u[0]

    blk_val = None
    if blk_col:
        u2 = pd.unique(blk_src[blk_col].dropna())
        if len(u2) >= 1:
            blk_val = u2[0]

    ctx.ids["cycle_index"] = cyc_val
    ctx.ids["block_id"] = blk_val


def _compute_basis(ctx: CycleContext) -> None:
    """
    Compute once per cycle; cache Q/E for charge & discharge, plus flags.
    Expects raw columns: time_s, current_a, voltage_v, step_type
    """
    required = {"time_s", "current_a", "voltage_v", "step_type"}
    missing = required - set(ctx.raw.columns)
    if missing:
        raise KeyError(f"raw_cycle_df missing required columns: {missing}")

    # Integrated from raw
    Qc, Ec, statsC = _sum_over_contiguous_runs(ctx.raw, "charge")
    Qd, Ed, statsD = _sum_over_contiguous_runs(ctx.raw, "discharge")

    # Device-measured step Ah (optional)
    Qc_meas, statsC_meas = _sum_measured_step_ah_over_runs(ctx.raw, "charge")
    Qd_meas, statsD_meas = _sum_measured_step_ah_over_runs(ctx.raw, "discharge")

    ctx.cache.update({
        "Q_charge_Ah": Qc,
        "E_charge_Wh": Ec,
        "Q_discharge_Ah": Qd,
        "E_discharge_Wh": Ed,
        "Q_charge_Ah_meas": Qc_meas,              # may be NaN if column missing
        "Q_discharge_Ah_meas": Qd_meas,           # may be NaN if column missing
        "has_charge": statsC["runs"] > 0,
        "has_discharge": statsD["runs"] > 0,
        "is_partial_cycle": not (statsC["runs"] > 0 and statsD["runs"] > 0),
        "debug_stats": {
            "charge":    {**statsC, "meas_runs": statsC_meas["runs"], "meas_rows": statsC_meas["rows"]},
            "discharge": {**statsD, "meas_runs": statsD_meas["runs"], "meas_rows": statsD_meas["rows"]},
        },
    })


# -----------------------
# Minimal KPI units
# -----------------------

@dataclass
class QChargeAh:
    name: str = "Q_charge_Ah"
    requires_basis: Tuple[str, ...] = ("Q_charge_Ah",)

    def compute(self, ctx: CycleContext) -> Dict[str, Any]:
        return {self.name: float(ctx.cache["Q_charge_Ah"])}

@dataclass
class QDischargeAh:
    name: str = "Q_discharge_Ah"
    requires_basis: Tuple[str, ...] = ("Q_discharge_Ah",)

    def compute(self, ctx: CycleContext) -> Dict[str, Any]:
        return {self.name: float(ctx.cache["Q_discharge_Ah"])}

@dataclass
class EChargeWh:
    name: str = "E_charge_Wh"
    requires_basis: Tuple[str, ...] = ("E_charge_Wh",)

    def compute(self, ctx: CycleContext) -> Dict[str, Any]:
        return {self.name: float(ctx.cache["E_charge_Wh"])}

@dataclass
class EDischargeWh:
    name: str = "E_discharge_Wh"
    requires_basis: Tuple[str, ...] = ("E_discharge_Wh",)

    def compute(self, ctx: CycleContext) -> Dict[str, Any]:
        return {self.name: float(ctx.cache["E_discharge_Wh"])}

@dataclass
class CoulombicEfficiency:
    """
    CE (integrated) = Q_discharge_Ah / Q_charge_Ah
    Returns NaN if either side missing or Q_charge_Ah == 0.
    """
    name: str = "CoulombicEfficiency"
    requires_basis: Tuple[str, ...] = ("Q_charge_Ah", "Q_discharge_Ah")

    def compute(self, ctx: CycleContext) -> Dict[str, Any]:
        Qc = ctx.cache["Q_charge_Ah"]
        Qd = ctx.cache["Q_discharge_Ah"]
        if ctx.cache["has_charge"] and ctx.cache["has_discharge"] and Qc > 0:
            return {self.name: float(Qd / Qc)}
        return {self.name: float("nan")}

@dataclass
class CoulombicEfficiencyMeasured:
    """
    CE (measured) = Q_discharge_Ah_meas / Q_charge_Ah_meas
    Uses device 'step_charge_ah'; returns NaN if unavailable.
    """
    name: str = "CoulombicEfficiency_meas"
    requires_basis: Tuple[str, ...] = ("Q_charge_Ah_meas", "Q_discharge_Ah_meas")

    def compute(self, ctx: CycleContext) -> Dict[str, Any]:
        Qc = ctx.cache.get("Q_charge_Ah_meas", float("nan"))
        Qd = ctx.cache.get("Q_discharge_Ah_meas", float("nan"))
        if np.isfinite(Qc) and np.isfinite(Qd) and Qc > 0:
            return {self.name: float(Qd / Qc)}
        return {self.name: float("nan")}


def compute_single_cycle_kpis(
    raw_cycle_df: pd.DataFrame,
    doe_cycle_df: pd.DataFrame,
    kpi_units: List[SingleCycleKPI],
) -> Dict[str, Any]:
    """
    Compute requested KPIs for ONE selected cycle.
    Input:
      - raw_cycle_df: rows for exactly one cycle (contains charge/OCV/discharge)
      - doe_cycle_df: DOE rows for the same cycle (to read cycle/block ids if present)
      - kpi_units: e.g., [QChargeAh(), QDischargeAh(), EChargeWh(), EDischargeWh(), CoulombicEfficiency(), ...]
    Output dict includes:
      block_id, cycle_index, requested KPIs, flags (has_charge, has_discharge, is_partial_cycle), debug_stats
    """
    ctx = CycleContext(raw=raw_cycle_df.copy(), doe=doe_cycle_df.copy())
    _extract_ids(ctx)
    _compute_basis(ctx)

    out: Dict[str, Any] = {
        "block_id": ctx.ids.get("block_id"),
        "cycle_index": ctx.ids.get("cycle_index"),
        "has_charge": ctx.cache["has_charge"],
        "has_discharge": ctx.cache["has_discharge"],
        "is_partial_cycle": ctx.cache["is_partial_cycle"],
        "debug_stats": ctx.cache["debug_stats"],
    }

    for unit in kpi_units:
        missing = [b for b in unit.requires_basis if b not in ctx.cache]
        if missing:
            raise RuntimeError(f"KPI '{unit.name}' missing basis {missing}. "
                               f"Available: {list(ctx.cache.keys())}")
        out.update(unit.compute(ctx))

    return out


# ======================================================================
# Multi-cycle orchestrator: runs the single-cycle engine across many cycles
# ======================================================================

def compute_multi_cycle_kpi_table(
    raw_df: pd.DataFrame,
    doe_df: pd.DataFrame,
    kpi_units: List[SingleCycleKPI],
    cycles: Optional[List[Any]] = None,                 # optional subset of cycle ids
    block_col: str = "block_id",
    cycle_col_candidates: Tuple[str, ...] = ("cycle_number", "cycle_index"),
    step_col: str = "step_number",
) -> pd.DataFrame:
    """
    DOE-driven multi-cycle KPI orchestrator.

    - Enumerates cycles from DOE using the first available cycle column
      in `cycle_col_candidates`.
    - For each cycle, collects DOE step numbers and slices RAW by
      `step_number` (and, if available, same block_id).
    - Runs the single-cycle KPI engine and aggregates results.

    Assumptions:
      * RAW has 'step_number' so we can map DOE->RAW.
      * Single block in this run (but function optionally filters by block_id if present).
    """
    # 1) Find cycle column in DOE
    cyc_col = next((c for c in cycle_col_candidates if c in doe_df.columns), None)
    if cyc_col is None:
        raise KeyError(
            f"No cycle id column found in DOE (looked for {cycle_col_candidates}). "
            "When cycles are DOE-defined, we must read them from DOE."
        )

    if step_col not in doe_df.columns or step_col not in raw_df.columns:
        raise KeyError(
            f"'step_number' mapping required. Missing in: "
            f"{'DOE ' if step_col not in doe_df.columns else ''}"
            f"{'RAW' if step_col not in raw_df.columns else ''}"
        )

    # 2) Enumerate cycles to process
    all_cycles = pd.unique(doe_df[cyc_col].dropna())
    if cycles is not None:
        wanted = set(cycles)
        all_cycles = [c for c in all_cycles if c in wanted]

    rows = []
    for c in all_cycles:
        doe_cycle = doe_df[doe_df[cyc_col] == c]

        # Gather step numbers for this cycle from DOE
        step_ids = pd.unique(doe_cycle[step_col].dropna())
        if len(step_ids) == 0:
            # No steps in DOE for this cycle -> emit empty/flagged row?
            # Here we skip; alternatively, append a flagged row.
            continue

        # Slice RAW by step_number (and optionally by block_id)
        raw_slice = raw_df[raw_df[step_col].isin(step_ids)].copy()

        # If both RAW and DOE have block_id, keep only that block in RAW to be safe
        if block_col in doe_cycle.columns and block_col in raw_slice.columns:
            blk_val = doe_cycle[block_col].iloc[0]
            raw_slice = raw_slice[raw_slice[block_col] == blk_val]

        # Run single-cycle KPIs
        res = compute_single_cycle_kpis(raw_slice, doe_cycle, kpi_units)

        # Ensure IDs explicitly from DOE
        if "cycle_index" not in res or pd.isna(res["cycle_index"]):
            res["cycle_index"] = c
        # also include cycle_number for convenience if cyc_col is that
        if cyc_col == "cycle_number":
            res["cycle_number"] = c

        if ("block_id" not in res or pd.isna(res["block_id"])) and (block_col in doe_cycle.columns):
            res["block_id"] = doe_cycle[block_col].iloc[0]

        rows.append(res)

    return pd.DataFrame(rows)


# ======================================================================
# Meta-KPI: add capacity retention (%) per block vs first valid discharge
# ======================================================================
def add_capacity_retention(
    cycle_kpi_df: pd.DataFrame,
    discharge_col: str = "Q_discharge_Ah",
    block_col: str = "block_id",
    out_col: str = "CapacityRetention_%",
    cycle_col: str = "cycle_index",  # used only for ordering baseline selection if present
) -> pd.DataFrame:
    """
    Adds CapacityRetention_% = 100 * Q_discharge_Ah / (first valid Q_discharge_Ah),
    computed per block if block_col exists, else globally.

    Robust, no groupby.apply; uses a map of baselines per block.
    """
    df = cycle_kpi_df.copy()

    if discharge_col not in df.columns:
        raise KeyError(f"'{discharge_col}' not found in cycle KPI table.")

    q = pd.to_numeric(df[discharge_col], errors="coerce")

    def first_positive_per_block(g: pd.DataFrame) -> float:
        # sort by cycle if present to define "first"
        if cycle_col in g.columns:
            g = g.sort_values(cycle_col)
        qg = pd.to_numeric(g[discharge_col], errors="coerce")
        pos = qg[qg > 0]
        return float(pos.iloc[0]) if len(pos) else np.nan

    if block_col in df.columns:
        # compute baseline q0 per block
        order_cols = [block_col]
        if cycle_col in df.columns:
            order_cols.append(cycle_col)
        tmp = df.sort_values(order_cols)
        first_q0 = (
            tmp[tmp[discharge_col].apply(pd.to_numeric, errors="coerce") > 0]
            .groupby(block_col, sort=False)[discharge_col]
            .first()
            .astype(float)
        )
        # map baseline back to rows
        q0 = df[block_col].map(first_q0)
    else:
        # single-block run or no block column
        if cycle_col in df.columns:
            tmp = df.sort_values(cycle_col)
        else:
            tmp = df
        pos = pd.to_numeric(tmp[discharge_col], errors="coerce")
        pos = pos[pos > 0]
        q0_scalar = float(pos.iloc[0]) if len(pos) else np.nan
        q0 = pd.Series(q0_scalar, index=df.index)

    # compute retention and mask invalid rows
    retention = 100.0 * q / q0
    retention[~(q > 0)] = np.nan  # no discharge -> no retention

    df[out_col] = retention
    return df
