
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Protocol, Optional
import numpy as np
import pandas as pd

# =================================================================================
# Helper Functions 
# =================================================================================


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

def _sum_duration_by_mask(raw: pd.DataFrame, mask: pd.Series) -> float:
    """
    Sum duration in seconds over rows where `mask` is True.
    Robust to sorting: we realign the mask to the sorted frame by index.
    """
    if raw.empty:
        return 0.0

    w = raw[["time_s"]].dropna().copy()
    w = w.sort_values("time_s")
    # REALIGN the mask to the sorted index (avoid mask.values!)
    m = mask.reindex(w.index).fillna(False).astype(bool).to_numpy()

    if len(w) < 2:
        return 0.0

    t = w["time_s"].to_numpy()
    dt = np.diff(t)
    valid = (dt > 0) & m[:-1] & m[1:]
    return float(np.sum(dt[valid]))


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


def _compute_basis(ctx: CycleContext) -> None:
    """
    Compute once per cycle and cache core basis + CC/CV-aware features.

    Core basis (from raw, robust trapezoids):
      - Q_charge_Ah, E_charge_Wh, Q_discharge_Ah, E_discharge_Wh
      - flags: has_charge, has_discharge, is_partial_cycle
      - debug_stats: runs/rows/pairs per step_type
      - measured device sums (optional): Q_charge_Ah_meas, Q_discharge_Ah_meas

    CC/CV-aware features (if 'cc_cv' exists):
      - Charge_total_duration_s, Discharge_total_duration_s
      - Charge_CV_duration_s
      - Q_charge_Ah_CV, Q_charge_Ah_CC
      - Q_charge_Ah_by_cccv (denominator for ratios = CV+CC capacity)
      - cccv_vs_total_mismatch_% (diagnostic vs total charge Q)
    """
    # --- Required raw columns for core basis ---
    required = {"time_s", "current_a", "voltage_v", "step_type"}
    missing = required - set(ctx.raw.columns)
    if missing:
        raise KeyError(f"raw_cycle_df missing required columns: {missing}")

    # === Core basis: integrated from raw by step_type ===
    Qc, Ec, statsC = _sum_over_contiguous_runs(ctx.raw, "charge")
    Qd, Ed, statsD = _sum_over_contiguous_runs(ctx.raw, "discharge")

    # Device-measured step Ah (optional)
    Qc_meas, statsC_meas = _sum_measured_step_ah_over_runs(ctx.raw, "charge")
    Qd_meas, statsD_meas = _sum_measured_step_ah_over_runs(ctx.raw, "discharge")

    has_charge = statsC["runs"] > 0
    has_discharge = statsD["runs"] > 0
    is_partial_cycle = not (has_charge and has_discharge)

    debug_stats = {
        "charge":    {**statsC, "meas_runs": statsC_meas["runs"], "meas_rows": statsC_meas["rows"]},
        "discharge": {**statsD, "meas_runs": statsD_meas["runs"], "meas_rows": statsD_meas["rows"]},
    }

    # === Defaults for CC/CV-aware entries ===
    charge_total_dur_s = float("nan")
    discharge_total_dur_s = float("nan")
    charge_cv_dur_s = float("nan")
    q_charge_cv = float("nan")
    q_charge_cc = float("nan")
    q_charge_by_cccv = float("nan")
    mismatch_pct = float("nan")

    # === Compute durations by step_type (time_s + step_type) ===
    if {"time_s", "step_type"}.issubset(ctx.raw.columns):
        step_lc = ctx.raw["step_type"].astype(str).str.lower()
        charge_total_dur_s = _sum_duration_by_mask(ctx.raw, step_lc.eq("charge"))
        discharge_total_dur_s = _sum_duration_by_mask(ctx.raw, step_lc.eq("discharge"))

    # === CC/CV metrics only if 'cc_cv' exists ===
    has_cc_cv_col = "cc_cv" in ctx.raw.columns
    if has_cc_cv_col:
        step_lc = ctx.raw["step_type"].astype(str).str.lower()
        mode_uc = ctx.raw["cc_cv"].astype(str).str.strip().str.upper()

        mask_charge_cv = step_lc.eq("charge") & mode_uc.eq("CV")
        mask_charge_cc = step_lc.eq("charge") & mode_uc.eq("CC")

        # CV duration during charge (robust mask alignment)
        charge_cv_dur_s = _sum_duration_by_mask(ctx.raw, mask_charge_cv)

        # Capacities by partition (subset raw and reuse integrator)
        sub_cv = ctx.raw.loc[mask_charge_cv]
        sub_cc = ctx.raw.loc[mask_charge_cc]

        if not sub_cv.empty and {"time_s","current_a","voltage_v"}.issubset(sub_cv.columns):
            q_charge_cv, _, _, _ = _integrate_trapz_abs(sub_cv)
        if not sub_cc.empty and {"time_s","current_a","voltage_v"}.issubset(sub_cc.columns):
            q_charge_cc, _, _, _ = _integrate_trapz_abs(sub_cc)

        # Partition-consistent denominator
        denom = (q_charge_cv if np.isfinite(q_charge_cv) else 0.0) + \
                (q_charge_cc if np.isfinite(q_charge_cc) else 0.0)
        q_charge_by_cccv = denom if denom > 0 else float("nan")

        # Diagnostic vs total charge capacity (use local Qc, not cache)
        if np.isfinite(Qc) and Qc > 0 and np.isfinite(denom):
            mismatch_pct = 100.0 * abs(denom - Qc) / Qc

    # === Write everything to cache (single update) ===
    ctx.cache.update({
        # Core basis
        "Q_charge_Ah": Qc,
        "E_charge_Wh": Ec,
        "Q_discharge_Ah": Qd,
        "E_discharge_Wh": Ed,
        "Q_charge_Ah_meas": Qc_meas,        # may be NaN if column missing
        "Q_discharge_Ah_meas": Qd_meas,     # may be NaN if column missing
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


# ----------------------------
# Duration KPIs
# ----------------------------

@dataclass
class ChargeCVDurationSeconds:
    """
    Charge_CV_duration_s: total time spent in CV during charge (seconds).

    Calculation:
      Sum of positive time deltas (dt) across rows where:
          step_type == 'charge' AND cc_cv == 'CV'
      (computed robustly in the basis layer).

    Required columns (raw):
      - time_s, step_type, cc_cv

    Purpose:
      Tracks how much of the charge step is governed by constant voltage,
      often increasing as the cell polarizes (useful aging indicator).
    """
    name: str = "Charge_CV_duration_s"
    requires_basis: Tuple[str, ...] = ("Charge_CV_duration_s",)

    def compute(self, ctx: CycleContext) -> Dict[str, Any]:
        return {self.name: float(ctx.cache["Charge_CV_duration_s"])}


@dataclass
class ChargeTotalDurationSeconds:
    """
    Charge_total_duration_s: total charge-step time (seconds).

    Calculation:
      Sum of positive dt across rows where step_type == 'charge'.

    Required columns (raw):
      - time_s, step_type

    Purpose:
      Baseline for time-ratio KPIs; helps diagnose rate changes or pauses.
    """
    name: str = "Charge_total_duration_s"
    requires_basis: Tuple[str, ...] = ("Charge_total_duration_s",)

    def compute(self, ctx: CycleContext) -> Dict[str, Any]:
        return {self.name: float(ctx.cache["Charge_total_duration_s"])}

# ----------------------------
# Ratios (time & capacity)
# ----------------------------

@dataclass
class ChargeCVTimeRatio:
    """
    Charge_CV_time_ratio: fraction of charge time spent in CV (0..1).

    Calculation:
      Charge_CV_duration_s / Charge_total_duration_s

    Required basis:
      - Charge_CV_duration_s
      - Charge_total_duration_s

    Purpose:
      Time-based view of how dominant the CV phase is during charging.
      Useful for tracking impedance/polarization drift across cycles.
    """
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
    Charge_CV_capacity_ratio (0..1):
      = Q_charge_Ah_CV / (Q_charge_Ah_CV + Q_charge_Ah_CC)
    Falls back to Q_charge_Ah total if CC/CV split is unavailable.

    Purpose:
      Capacity-based view of CV dominance, robust to timing glitches that
      could bias the whole-charge integral.
    """
    name: str = "Charge_CV_capacity_ratio"
    requires_basis: Tuple[str, ...] = ("Q_charge_Ah_CV", "Q_charge_Ah_by_cccv")

    def compute(self, ctx: CycleContext) -> Dict[str, Any]:
        qcv = ctx.cache["Q_charge_Ah_CV"]
        denom = ctx.cache["Q_charge_Ah_by_cccv"]

        # Fallback if split unavailable
        if not np.isfinite(denom) or denom <= 0:
            qt = ctx.cache.get("Q_charge_Ah", float("nan"))
            val = float(qcv / qt) if (np.isfinite(qt) and qt > 0) else float("nan")
            return {self.name: val}

        val = float(qcv / denom) if (np.isfinite(qcv) and denom > 0) else float("nan")
        return {self.name: val}

@dataclass
class CCVTotalMismatchPct:
    """Diagnostic: |(Q_CV + Q_CC) - Q_total| / Q_total * 100 [%]"""
    name: str = "cccv_vs_total_mismatch_%"
    requires_basis: Tuple[str, ...] = ("cccv_vs_total_mismatch_%",)

    def compute(self, ctx: CycleContext) -> Dict[str, Any]:
        return {self.name: float(ctx.cache["cccv_vs_total_mismatch_%"])}


# ----------------------------
# Energy efficiency & average voltages
# ----------------------------

@dataclass
class EnergyEfficiency:
    """
    EnergyEfficiency: energy ratio (0..~1+ε) per cycle.

    Calculation:
      E_discharge_Wh / E_charge_Wh
      (NaN if E_charge_Wh == 0 or either side missing.)

    Required basis:
      - E_charge_Wh
      - E_discharge_Wh

    Purpose:
      Quantifies round-trip energy efficiency; sensitive to polarization,
      IR drop, and kinetic losses. Expect values near 1 for healthy cells,
      slightly <1 due to losses; minor >1 can occur from noise/sampling.
    """
    name: str = "EnergyEfficiency"
    requires_basis: Tuple[str, ...] = ("E_charge_Wh", "E_discharge_Wh")

    def compute(self, ctx: CycleContext) -> Dict[str, Any]:
        ec = ctx.cache["E_charge_Wh"]
        ed = ctx.cache["E_discharge_Wh"]
        val = float(ed / ec) if (isinstance(ec, (int, float)) and ec > 0) else float("nan")
        return {self.name: val}


@dataclass
class VChargeAvg:
    """
    V_charge_avg: average voltage during charge (V).

    Calculation:
      E_charge_Wh / Q_charge_Ah
      (NaN if Q_charge_Ah == 0)

    Required basis:
      - E_charge_Wh
      - Q_charge_Ah

    Purpose:
      Compact proxy for polarization changes; increasing average charge
      voltage across cycles can signal rising impedance.
    """
    name: str = "V_charge_avg"
    requires_basis: Tuple[str, ...] = ("E_charge_Wh", "Q_charge_Ah")

    def compute(self, ctx: CycleContext) -> Dict[str, Any]:
        q = ctx.cache["Q_charge_Ah"]
        e = ctx.cache["E_charge_Wh"]
        val = float(e / q) if (isinstance(q, (int, float)) and q > 0) else float("nan")
        return {self.name: val}


@dataclass
class VDischargeAvg:
    """
    V_discharge_avg: average voltage during discharge (V).

    Calculation:
      E_discharge_Wh / Q_discharge_Ah
      (NaN if Q_discharge_Ah == 0)

    Required basis:
      - E_discharge_Wh
      - Q_discharge_Ah

    Purpose:
      Tracks discharge polarization; decreasing values can indicate
      increasing internal resistance or degradation.
    """
    name: str = "V_discharge_avg"
    requires_basis: Tuple[str, ...] = ("E_discharge_Wh", "Q_discharge_Ah")

    def compute(self, ctx: CycleContext) -> Dict[str, Any]:
        q = ctx.cache["Q_discharge_Ah"]
        e = ctx.cache["E_discharge_Wh"]
        val = float(e / q) if (isinstance(q, (int, float)) and q > 0) else float("nan")
        return {self.name: val}


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

def add_energy_retention(
    cycle_kpi_df: pd.DataFrame,
    energy_col: str = "E_discharge_Wh",
    block_col: str = "block_id",
    out_col: str = "EnergyRetention_%",
    cycle_col: str = "cycle_index",
) -> pd.DataFrame:
    """
    Add EnergyRetention_% to a cycle KPI table.

    Definition:
      EnergyRetention_% = 100 * E_discharge_Wh / E_discharge_Wh_baseline

    Baseline policy:
      For each block (if present), the baseline is the **first** cycle with
      positive E_discharge_Wh after sorting by `cycle_index` when available.
      If no block column exists, a single global baseline is used.

    Required columns:
      - energy_col (default 'E_discharge_Wh')
      - Optional: block_col (default 'block_id')
      - Optional: cycle_col (default 'cycle_index') for ordering the baseline

    Purpose:
      Tracks retention of delivered energy over cycling, complementing
      capacity retention for cases where voltage evolution matters.
    """
    df = cycle_kpi_df.copy()

    if energy_col not in df.columns:
        raise KeyError(f"'{energy_col}' not found in cycle KPI table.")

    e = pd.to_numeric(df[energy_col], errors="coerce")

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
        if cycle_col in df.columns:
            tmp = df.sort_values(cycle_col)
        else:
            tmp = df
        pos = pd.to_numeric(tmp[energy_col], errors="coerce")
        pos = pos[pos > 0]
        e0_scalar = float(pos.iloc[0]) if len(pos) else np.nan
        e0 = pd.Series(e0_scalar, index=df.index)

    ret = 100.0 * e / e0
    ret[~(e > 0)] = np.nan  # no discharge energy -> no retention
    df[out_col] = ret
    return df
