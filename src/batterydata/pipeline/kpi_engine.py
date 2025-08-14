from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Protocol, Optional
import numpy as np
import pandas as pd

"""
KPI CALCULATOR 
===========================

Purpose
-------
Compute per-cycle KPIs (capacity, energy, efficiencies, durations, CC/CV ratios)
from RAW and DOE tables for battery cycling experiments.

Key Fixes vs. Previous Version
------------------------------
1) **CC/CV capacity integration is now "contiguous-run safe"**:
   We integrate CV and CC **per contiguous run** of (step_type, cc_cv).
   This prevents bridging over gaps when CV segments are non-contiguous, which
   previously inflated CV capacity and distorted ratios.

2) **One-pass normalization and sorting**:
   We sort by `time_s` once and normalize `step_type`/`cc_cv` once, avoiding
   repeated `.copy()`, casing, and sorting overhead across helpers.

3) **Clearer documentation and naming**:
   Docstrings specify units, sign conventions, and assumptions.

Design Notes
------------
- Energy and charge magnitudes are computed as **positive** numbers using |I|
  and |I*V| integrals. This aligns with KPI magnitude comparisons (e.g., CE,
  energy efficiency).
- The code trusts `step_type` and `cc_cv` labels provided upstream. Durations
  are robust (mask-based). Capacities by CC/CV are robust to multi-run cases
  (contiguous-run grouping) but will, by design, ignore unlabeled rows.
- We expose both the **CV/CC capacities** and the **mismatch diagnostic** to
  allow QC against total charge.

Public API 
----------------------
- compute_single_cycle_kpis(raw_cycle_df, doe_cycle_df, kpi_units)
- compute_multi_cycle_kpi_table(raw_df, doe_df, kpi_units, ...)
- add_capacity_retention(...), add_energy_retention(...)

Additional KPI Units
--------------------
- QChargeAhCV, QChargeAhCC, QChargeAhByCCCV (so CV/CC magnitudes can be added
  to the KPI table explicitly).
"""

# =================================================================================
# Helper Functions (low-level, fast, side-effect free)
# =================================================================================

def _ensure_sorted(df: pd.DataFrame, by: str = "time_s") -> pd.DataFrame:
    """Return df sorted by `by` (stable sort), no copy if already sorted.

    Assumes `by` exists. If input is small or unsorted, the cost is negligible
    relative to safety of ensuring monotonic time for integration.
    """
    if df.empty:
        return df
    # Fast path: check monotonic. If not monotonic increasing, sort.
    s = df[by]
    if not s.is_monotonic_increasing:
        return df.sort_values(by, kind="mergesort")
    return df


def _integrate_trapz_abs_sorted(df: pd.DataFrame) -> Tuple[float, float, int, int]:
    """
    Integrate |I| and |I*V| over strictly positive dt using trapezoids.

    Preconditions
    -------------
    - df is already sorted by `time_s` ascending.
    - Columns present: time_s [s], current_a [A], voltage_v [V].

    Returns
    -------
    (Q_Ah, E_Wh, n_rows_used, n_pairs_used)
    where Q_Ah = ∫|I| dt / 3600 and E_Wh = ∫|I·V| dt / 3600.
    """
    needed = {"time_s", "current_a", "voltage_v"}
    if not needed.issubset(df.columns):
        raise KeyError(f"Missing columns for integration: {needed - set(df.columns)}")

    w = df[["time_s", "current_a", "voltage_v"]].dropna()
    if len(w) < 2:
        return 0.0, 0.0, int(len(w)), 0

    t = w["time_s"].to_numpy()
    I = np.abs(w["current_a"].to_numpy())
    V = w["voltage_v"].to_numpy()

    dt = np.diff(t)
    mask = dt > 0
    if not np.any(mask):
        return 0.0, 0.0, int(len(w)), 0

    # Trapezoid midpoints
    I_mid = 0.5 * (I[:-1] + I[1:])
    IV_mid = 0.5 * (np.abs(I[:-1] * V[:-1]) + np.abs(I[1:] * V[1:]))

    Q_As = np.sum(I_mid[mask] * dt[mask])   # [A·s]
    E_Ws = np.sum(IV_mid[mask] * dt[mask])  # [W·s]

    return float(Q_As / 3600.0), float(E_Ws / 3600.0), int(len(w)), int(np.sum(mask))


def _sum_duration_by_mask_sorted(time_s: pd.Series, mask_sorted: pd.Series) -> float:
    """Sum duration [s] over rows where `mask_sorted` is True, assuming `time_s`
    is sorted ascending and `mask_sorted` is index-aligned.

    We count only dt where **both** endpoints satisfy the mask (no bridging).
    """
    if len(time_s) < 2:
        return 0.0
    t = time_s.to_numpy()
    dt = np.diff(t)
    m = mask_sorted.to_numpy(dtype=bool)
    valid = (dt > 0) & m[:-1] & m[1:]
    return float(np.sum(dt[valid]))


# =================================================================================
# Contiguous-run aggregators
# =================================================================================

def _sum_over_step_runs_sorted(df_sorted: pd.DataFrame, target_step: str) -> Tuple[float, float, Dict[str, int]]:
    """Integrate Q/E over **contiguous runs** where `step_type == target_step`.

    Requires df_sorted columns: time_s, current_a, voltage_v, _step_lc, _run_step
    (created by `_prepare_for_basis`).
    """
    if df_sorted.empty:
        return 0.0, 0.0, {"runs": 0, "rows": 0, "pairs": 0}

    total_Q = total_E = 0.0
    runs = rows = pairs = 0
    tgt = target_step.lower()

    for _, sub in df_sorted.groupby("_run_step", sort=True):
        if sub["_step_lc"].iat[0] != tgt:
            continue
        Q, E, n_rows, n_pairs = _integrate_trapz_abs_sorted(sub)
        total_Q += Q
        total_E += E
        runs += 1
        rows += n_rows
        pairs += n_pairs

    return total_Q, total_E, {"runs": runs, "rows": rows, "pairs": pairs}


def _sum_measured_step_ah_over_step_runs_sorted(df_sorted: pd.DataFrame, target_step: str) -> Tuple[float, Dict[str, int]]:
    """Sum device-integrated step charge (Ah) over contiguous runs where
    `step_type == target_step` using `step_charge_ah`.

    Robust to non-zero start by using |last - first| per run.
    Returns (Q_meas_Ah, stats). If column missing, returns (NaN, {...}).
    """
    if "step_charge_ah" not in df_sorted.columns:
        return float("nan"), {"runs": 0, "rows": 0}
    if df_sorted.empty:
        return 0.0, {"runs": 0, "rows": 0}

    total_Q = 0.0
    runs = rows = 0
    tgt = target_step.lower()

    for _, sub in df_sorted.groupby("_run_step", sort=True):
        if sub["_step_lc"].iat[0] != tgt:
            continue
        sub2 = sub[["time_s", "step_charge_ah"]].dropna()
        if len(sub2) == 0:
            continue
        q_first = float(sub2["step_charge_ah"].iat[0])
        q_last = float(sub2["step_charge_ah"].iat[-1])
        total_Q += abs(q_last - q_first)
        runs += 1
        rows += len(sub2)

    return float(total_Q), {"runs": runs, "rows": rows}


def _sum_over_charge_cccv_runs_sorted(df_sorted: pd.DataFrame) -> Tuple[float, float, Dict[str, int]]:
    """Integrate Q (Ah) for charge-time **CV** and **CC** **per contiguous run**.

    Returns
    -------
    (Q_CV_Ah, Q_CC_Ah, stats_dict)

    `stats_dict` fields:
      - runs_cv, runs_cc: number of contiguous runs integrated for CV and CC
      - rows_cv, rows_cc: total rows used across those runs
      - pairs_cv, pairs_cc: total positive-dt trapezoid pairs across those runs
    """
    if df_sorted.empty:
        return float("nan"), float("nan"), {
            "runs_cv": 0, "rows_cv": 0, "pairs_cv": 0,
            "runs_cc": 0, "rows_cc": 0, "pairs_cc": 0,
        }

    Qcv = Qcc = 0.0
    runs_cv = rows_cv = pairs_cv = 0
    runs_cc = rows_cc = pairs_cc = 0

    for _, sub in df_sorted.groupby("_run_both", sort=True):
        st = sub["_step_lc"].iat[0]
        ccv = sub["_cccv_uc"].iat[0]
        if st != "charge":
            continue
        if ccv == "CV":
            Q, _, n_rows, n_pairs = _integrate_trapz_abs_sorted(sub)
            Qcv += Q
            runs_cv += 1
            rows_cv += n_rows
            pairs_cv += n_pairs
        elif ccv == "CC":
            Q, _, n_rows, n_pairs = _integrate_trapz_abs_sorted(sub)
            Qcc += Q
            runs_cc += 1
            rows_cc += n_rows
            pairs_cc += n_pairs
        # else: ignore unlabeled/other

    stats = {
        "runs_cv": runs_cv, "rows_cv": rows_cv, "pairs_cv": pairs_cv,
        "runs_cc": runs_cc, "rows_cc": rows_cc, "pairs_cc": pairs_cc,
    }
    # return NaN if nothing was found at all
    if runs_cv == 0 and runs_cc == 0:
        return float("nan"), float("nan"), stats
    return float(Qcv), float(Qcc), stats


# =================================================================================
# Context and basis computation
# =================================================================================

class SingleCycleKPI(Protocol):
    """A KPI unit that uses the single-cycle context and returns a dict."""
    name: str
    requires_basis: Tuple[str, ...]

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


# ---- ID extraction ----

def _extract_ids(ctx: CycleContext) -> None:
    """Fill ctx.ids with cycle_index and block_id, preferring DOE; fallback RAW.

    Accepts 'cycle_index' or 'cycle_number' as cycle id. 'block_id' is optional.
    Picks the first unique value found in the given slice.
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


# ---- Basis computation ----

def _prepare_for_basis(raw: pd.DataFrame) -> pd.DataFrame:
    """Return a sorted copy of raw with normalized helper columns:

    Added columns
    -------------
    - `_step_lc`: lowercase step_type (str)
    - `_cccv_uc`: uppercase cc_cv (str; if missing, set to "")
    - `_run_step`: run id for contiguous step_type blocks
    - `_run_both`: run id for contiguous (step_type, cc_cv) blocks
    """
    required = {"time_s", "current_a", "voltage_v", "step_type"}
    missing = required - set(raw.columns)
    if missing:
        raise KeyError(f"raw_cycle_df missing required columns: {missing}")

    s = _ensure_sorted(raw, by="time_s").copy()

    step_lc = s["step_type"].astype(str).str.lower()
    if "cc_cv" in s.columns:
        cccv_uc = s["cc_cv"].astype(str).str.strip().str.upper()
    else:
        cccv_uc = pd.Series([""] * len(s), index=s.index)

    s["_step_lc"] = step_lc
    s["_cccv_uc"] = cccv_uc

    # Contiguous run ids
    s["_run_step"] = (s["_step_lc"] != s["_step_lc"].shift(1)).cumsum()
    s["_run_both"] = ((s["_step_lc"] != s["_step_lc"].shift(1)) | (s["_cccv_uc"] != s["_cccv_uc"].shift(1))).cumsum()

    return s


def _compute_basis(ctx: CycleContext) -> None:
    """Compute once per cycle and cache core basis + CC/CV-aware features.

    Core basis (from raw, magnitudes via trapezoids):
      - Q_charge_Ah, E_charge_Wh, Q_discharge_Ah, E_discharge_Wh
      - flags: has_charge, has_discharge, is_partial_cycle
      - debug_stats: runs/rows/pairs per step_type (+ measured runs/rows)
      - measured device sums (optional): Q_charge_Ah_meas, Q_discharge_Ah_meas

    CC/CV-aware features (if `cc_cv` exists):
      - Charge_total_duration_s, Discharge_total_duration_s
      - Charge_CV_duration_s
      - Q_charge_Ah_CV, Q_charge_Ah_CC (contiguous-run safe)
      - Q_charge_Ah_by_cccv = Q_CV + Q_CC (denominator for ratios)
      - cccv_vs_total_mismatch_% = |(Q_CV+Q_CC) - Q_total| / Q_total * 100
    """
    s = _prepare_for_basis(ctx.raw)

    # --- Core basis via contiguous step runs ---
    Qc, Ec, statsC = _sum_over_step_runs_sorted(s, "charge")
    Qd, Ed, statsD = _sum_over_step_runs_sorted(s, "discharge")

    Qc_meas, statsC_meas = _sum_measured_step_ah_over_step_runs_sorted(s, "charge")
    Qd_meas, statsD_meas = _sum_measured_step_ah_over_step_runs_sorted(s, "discharge")

    has_charge = statsC["runs"] > 0
    has_discharge = statsD["runs"] > 0
    is_partial_cycle = not (has_charge and has_discharge)

    debug_stats = {
        "charge":    {**statsC, "meas_runs": statsC_meas["runs"], "meas_rows": statsC_meas["rows"]},
        "discharge": {**statsD, "meas_runs": statsD_meas["runs"], "meas_rows": statsD_meas["rows"]},
    }

    # --- Durations by step_type ---
    charge_total_dur_s = float("nan")
    discharge_total_dur_s = float("nan")
    if {"time_s", "step_type"}.issubset(s.columns):
        step_lc = s["_step_lc"]
        mask_chg = step_lc.eq("charge")
        mask_dis = step_lc.eq("discharge")
        charge_total_dur_s = _sum_duration_by_mask_sorted(s["time_s"], mask_chg)
        discharge_total_dur_s = _sum_duration_by_mask_sorted(s["time_s"], mask_dis)

    # --- CC/CV metrics ---
    has_cc_cv_col = "cc_cv" in ctx.raw.columns
    charge_cv_dur_s = float("nan")
    q_charge_cv = float("nan")
    q_charge_cc = float("nan")
    q_charge_by_cccv = float("nan")
    mismatch_pct = float("nan")

    if has_cc_cv_col:
        step_lc = s["_step_lc"]
        ccv_uc  = s["_cccv_uc"]

        mask_charge_cv = step_lc.eq("charge") & ccv_uc.eq("CV")
        charge_cv_dur_s = _sum_duration_by_mask_sorted(s["time_s"], mask_charge_cv)

        # CV/CC capacities per contiguous run (fixes bridging)
        q_charge_cv, q_charge_cc, cc_stats = _sum_over_charge_cccv_runs_sorted(s)

        denom = 0.0
        if np.isfinite(q_charge_cv):
            denom += q_charge_cv
        if np.isfinite(q_charge_cc):
            denom += q_charge_cc
        q_charge_by_cccv = denom if denom > 0 else float("nan")

        if np.isfinite(Qc) and Qc > 0 and np.isfinite(q_charge_by_cccv):
            mismatch_pct = 100.0 * abs(q_charge_by_cccv - Qc) / Qc

        # Attach CC/CV run stats into debug block for visibility
        debug_stats["charge"].update({
            "cccv_runs_cv": cc_stats.get("runs_cv", 0),
            "cccv_runs_cc": cc_stats.get("runs_cc", 0),
            "cccv_pairs_cv": cc_stats.get("pairs_cv", 0),
            "cccv_pairs_cc": cc_stats.get("pairs_cc", 0),
        })

    # --- Cache write ---
    ctx.cache.update({
        # Core basis
        "Q_charge_Ah": Qc,
        "E_charge_Wh": Ec,
        "Q_discharge_Ah": Qd,
        "E_discharge_Wh": Ed,
        "Q_charge_Ah_meas": Qc_meas,        # may be NaN
        "Q_discharge_Ah_meas": Qd_meas,     # may be NaN
        "has_charge": has_charge,
        "has_discharge": has_discharge,
        "is_partial_cycle": is_partial_cycle,
        "debug_stats": debug_stats,

        # Durations and CC/CV-aware features
        "Charge_total_duration_s": charge_total_dur_s,
        "Discharge_total_duration_s": discharge_total_dur_s,
        "Charge_CV_duration_s": charge_cv_dur_s,
        "Q_charge_Ah_CV": q_charge_cv,
        "Q_charge_Ah_CC": q_charge_cc,
        "Q_charge_Ah_by_cccv": q_charge_by_cccv,
        "cccv_vs_total_mismatch_%": mismatch_pct,
        "has_cc_cv_column": has_cc_cv_col,
    })


# =================================================================================
# KPI Units (pluggable)
# =================================================================================

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
    """CE (integrated) = Q_discharge_Ah / Q_charge_Ah (NaN if Qc==0 or missing)."""
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
    """CE (measured) = Q_discharge_Ah_meas / Q_charge_Ah_meas (device counters)."""
    name: str = "CoulombicEfficiency_meas"
    requires_basis: Tuple[str, ...] = ("Q_charge_Ah_meas", "Q_discharge_Ah_meas")
    def compute(self, ctx: CycleContext) -> Dict[str, Any]:
        Qc = ctx.cache.get("Q_charge_Ah_meas", float("nan"))
        Qd = ctx.cache.get("Q_discharge_Ah_meas", float("nan"))
        if np.isfinite(Qc) and np.isfinite(Qd) and Qc > 0:
            return {self.name: float(Qd / Qc)}
        return {self.name: float("nan")}

# --- Duration KPIs ---

@dataclass
class ChargeCVDurationSeconds:
    name: str = "Charge_CV_duration_s"
    requires_basis: Tuple[str, ...] = ("Charge_CV_duration_s",)
    def compute(self, ctx: CycleContext) -> Dict[str, Any]:
        return {self.name: float(ctx.cache["Charge_CV_duration_s"])}

@dataclass
class ChargeTotalDurationSeconds:
    name: str = "Charge_total_duration_s"
    requires_basis: Tuple[str, ...] = ("Charge_total_duration_s",)
    def compute(self, ctx: CycleContext) -> Dict[str, Any]:
        return {self.name: float(ctx.cache["Charge_total_duration_s"])}

# --- Ratios (time & capacity) ---

@dataclass
class ChargeCVTimeRatio:
    name: str = "Charge_CV_time_ratio"
    requires_basis: Tuple[str, ...] = ("Charge_CV_duration_s", "Charge_total_duration_s")
    def compute(self, ctx: CycleContext) -> Dict[str, Any]:
        cv = ctx.cache["Charge_CV_duration_s"]
        total = ctx.cache["Charge_total_duration_s"]
        val = float(cv / total) if (isinstance(total, (int, float)) and total > 0) else float("nan")
        return {self.name: val}

@dataclass
class ChargeCVCapacityRatio:
    """
    Capacity-based CV fraction in [0..1].

    Definition
    ----------
    Default: Q_charge_Ah_CV / (Q_charge_Ah_CV + Q_charge_Ah_CC)
    Fallback: Q_charge_Ah_CV / Q_charge_Ah (when CV+CC is unavailable)
    """
    name: str = "Charge_CV_capacity_ratio"
    requires_basis: Tuple[str, ...] = ("Q_charge_Ah_CV", "Q_charge_Ah_by_cccv")
    def compute(self, ctx: CycleContext) -> Dict[str, Any]:
        qcv = ctx.cache["Q_charge_Ah_CV"]
        denom = ctx.cache["Q_charge_Ah_by_cccv"]
        if not np.isfinite(denom) or denom <= 0:
            qt = ctx.cache.get("Q_charge_Ah", float("nan"))
            val = float(qcv / qt) if (np.isfinite(qt) and qt > 0) else float("nan")
            return {self.name: val}
        val = float(qcv / denom) if (np.isfinite(qcv) and denom > 0) else float("nan")
        return {self.name: val}

@dataclass
class CCVTotalMismatchPct:
    name: str = "cccv_vs_total_mismatch_%"
    requires_basis: Tuple[str, ...] = ("cccv_vs_total_mismatch_%",)
    def compute(self, ctx: CycleContext) -> Dict[str, Any]:
        return {self.name: float(ctx.cache["cccv_vs_total_mismatch_%"])}

# --- Energy efficiency & average voltages ---

@dataclass
class EnergyEfficiency:
    name: str = "EnergyEfficiency"
    requires_basis: Tuple[str, ...] = ("E_charge_Wh", "E_discharge_Wh")
    def compute(self, ctx: CycleContext) -> Dict[str, Any]:
        ec = ctx.cache["E_charge_Wh"]
        ed = ctx.cache["E_discharge_Wh"]
        val = float(ed / ec) if (isinstance(ec, (int, float)) and ec > 0) else float("nan")
        return {self.name: val}

@dataclass
class VChargeAvg:
    name: str = "V_charge_avg"
    requires_basis: Tuple[str, ...] = ("E_charge_Wh", "Q_charge_Ah")
    def compute(self, ctx: CycleContext) -> Dict[str, Any]:
        q = ctx.cache["Q_charge_Ah"]
        e = ctx.cache["E_charge_Wh"]
        val = float(e / q) if (isinstance(q, (int, float)) and q > 0) else float("nan")
        return {self.name: val}

@dataclass
class VDischargeAvg:
    name: str = "V_discharge_avg"
    requires_basis: Tuple[str, ...] = ("E_discharge_Wh", "Q_discharge_Ah")
    def compute(self, ctx: CycleContext) -> Dict[str, Any]:
        q = ctx.cache["Q_discharge_Ah"]
        e = ctx.cache["E_discharge_Wh"]
        val = float(e / q) if (isinstance(q, (int, float)) and q > 0) else float("nan")
        return {self.name: val}

# --- NEW: expose CV/CC magnitudes directly as KPIs ---

@dataclass
class QChargeAhCV:
    name: str = "Q_charge_Ah_CV"
    requires_basis: Tuple[str, ...] = ("Q_charge_Ah_CV",)
    def compute(self, ctx: CycleContext) -> Dict[str, Any]:
        return {self.name: float(ctx.cache["Q_charge_Ah_CV"])}

@dataclass
class QChargeAhCC:
    name: str = "Q_charge_Ah_CC"
    requires_basis: Tuple[str, ...] = ("Q_charge_Ah_CC",)
    def compute(self, ctx: CycleContext) -> Dict[str, Any]:
        return {self.name: float(ctx.cache["Q_charge_Ah_CC"])}

@dataclass
class QChargeAhByCCCV:
    name: str = "Q_charge_Ah_by_cccv"
    requires_basis: Tuple[str, ...] = ("Q_charge_Ah_by_cccv",)
    def compute(self, ctx: CycleContext) -> Dict[str, Any]:
        return {self.name: float(ctx.cache["Q_charge_Ah_by_cccv"])}


# =================================================================================
# Single-cycle engine
# =================================================================================

def compute_single_cycle_kpis(
    raw_cycle_df: pd.DataFrame,
    doe_cycle_df: pd.DataFrame,
    kpi_units: List[SingleCycleKPI],
) -> Dict[str, Any]:
    """Compute requested KPIs for **one** selected cycle.

    Inputs
    ------
    raw_cycle_df : RAW rows for exactly one cycle (contains charge/OCV/discharge)
    doe_cycle_df : DOE rows for the same cycle (to read cycle/block ids if present)
    kpi_units    : e.g., [QChargeAh(), QDischargeAh(), ...]

    Output dict includes
    ---------------------
    block_id, cycle_index, requested KPIs,
    flags (has_charge, has_discharge, is_partial_cycle), debug_stats.
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
            raise RuntimeError(
                f"KPI '{unit.name}' missing basis {missing}. Available: {list(ctx.cache.keys())}"
            )
        out.update(unit.compute(ctx))

    return out


# =================================================================================
# Multi-cycle orchestrator
# =================================================================================

def compute_multi_cycle_kpi_table(
    raw_df: pd.DataFrame,
    doe_df: pd.DataFrame,
    kpi_units: List[SingleCycleKPI],
    cycles: Optional[List[Any]] = None,                 # optional subset of cycle ids
    block_col: str = "block_id",
    cycle_col_candidates: Tuple[str, ...] = ("cycle_number", "cycle_index"),
    step_col: str = "step_number",
) -> pd.DataFrame:
    """DOE-driven multi-cycle KPI orchestrator.

    - Enumerates cycles from DOE using the first available cycle column in
      `cycle_col_candidates`.
    - For each cycle, collects DOE step numbers and slices RAW by `step_number`
      (and, if available, same `block_id`).
    - Runs the single-cycle KPI engine and aggregates results.

    Assumptions
    -----------
    * RAW has `step_number` so we can map DOE→RAW.
    * If both RAW and DOE have `block_id`, we align them.
    """
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

    # Determine which cycles to process
    all_cycles = pd.unique(doe_df[cyc_col].dropna())
    if cycles is not None:
        wanted = set(cycles)
        all_cycles = [c for c in all_cycles if c in wanted]

    rows = []
    for c in all_cycles:
        doe_cycle = doe_df[doe_df[cyc_col] == c]

        step_ids = pd.unique(doe_cycle[step_col].dropna())
        if len(step_ids) == 0:
            continue

        raw_slice = raw_df[raw_df[step_col].isin(step_ids)].copy()

        if block_col in doe_cycle.columns and block_col in raw_slice.columns:
            blk_val = doe_cycle[block_col].iloc[0]
            raw_slice = raw_slice[raw_slice[block_col] == blk_val]

        res = compute_single_cycle_kpis(raw_slice, doe_cycle, kpi_units)

        if "cycle_index" not in res or pd.isna(res["cycle_index"]):
            res["cycle_index"] = c
        if cyc_col == "cycle_number":
            res["cycle_number"] = c
        if ("block_id" not in res or pd.isna(res["block_id"])) and (block_col in doe_cycle.columns):
            res["block_id"] = doe_cycle[block_col].iloc[0]

        rows.append(res)

    return pd.DataFrame(rows)


# =================================================================================
# Meta-KPIs: retention utilities
# =================================================================================

def add_capacity_retention(
    cycle_kpi_df: pd.DataFrame,
    discharge_col: str = "Q_discharge_Ah",
    block_col: str = "block_id",
    out_col: str = "CapacityRetention_%",
    cycle_col: str = "cycle_index",
) -> pd.DataFrame:
    """Add CapacityRetention_% = 100 * Q_discharge_Ah / (first positive Q_discharge_Ah),
    computed per block if `block_col` exists, else globally.
    """
    df = cycle_kpi_df.copy()
    if discharge_col not in df.columns:
        raise KeyError(f"'{discharge_col}' not found in cycle KPI table.")

    if block_col in df.columns:
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
        q0 = df[block_col].map(first_q0)
    else:
        tmp = df.sort_values(cycle_col) if cycle_col in df.columns else df
        pos = pd.to_numeric(tmp[discharge_col], errors="coerce")
        pos = pos[pos > 0]
        q0_scalar = float(pos.iloc[0]) if len(pos) else np.nan
        q0 = pd.Series(q0_scalar, index=df.index)

    q = pd.to_numeric(df[discharge_col], errors="coerce")
    retention = 100.0 * q / q0
    retention[~(q > 0)] = np.nan
    df[out_col] = retention
    return df


def add_energy_retention(
    cycle_kpi_df: pd.DataFrame,
    energy_col: str = "E_discharge_Wh",
    block_col: str = "block_id",
    out_col: str = "EnergyRetention_%",
    cycle_col: str = "cycle_index",
) -> pd.DataFrame:
    """Add EnergyRetention_% = 100 * E_discharge_Wh / baseline E_discharge_Wh.

    Baseline is the first cycle with positive E_discharge_Wh per block (if
    present) or globally otherwise.
    """
    df = cycle_kpi_df.copy()
    if energy_col not in df.columns:
        raise KeyError(f"'{energy_col}' not found in cycle KPI table.")

    if block_col in df.columns:
        order_cols = [block_col]
        if cycle_col in df.columns:
            order_cols.append(cycle_col)
        tmp = df.sort_values(order_cols)
        first_e0 = (
            tmp[tmp[energy_col].apply(pd.to_numeric, errors="coerce") > 0]
            .groupby(block_col, sort=False)[energy_col]
            .first()
            .astype(float)
        )
        e0 = df[block_col].map(first_e0)
    else:
        tmp = df.sort_values(cycle_col) if cycle_col in df.columns else df
        pos = pd.to_numeric(tmp[energy_col], errors="coerce")
        pos = pos[pos > 0]
        e0_scalar = float(pos.iloc[0]) if len(pos) else np.nan
        e0 = pd.Series(e0_scalar, index=df.index)

    e = pd.to_numeric(df[energy_col], errors="coerce")
    ret = 100.0 * e / e0
    ret[~(e > 0)] = np.nan
    df[out_col] = ret
    return df
