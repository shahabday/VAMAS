from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Protocol, Optional

import numpy as np
import pandas as pd

"""KPI calculator (DOE-driven, CC/CV‑safe)

This module computes per‑cycle battery KPIs from two input tables:

- RAW table: time series sampled during a test run. Must include at least
  `time_s`, `current_a`, `voltage_v`, `step_type`, and the foreign key
  `step_number`. Optional columns: `cc_cv` ("CC" / "CV"), `step_charge_ah`,
  `block_id`.

- DOE table: step‑level design-of-experiment descriptor. Must include
  `step_number` (to join to RAW) and `cycle_number` (the only cycle identifier
  used by this module). Optional: `block_id` (assumed unique in the provided DOE
  slice).

Important design choices
------------------------
- The module **never reads any cycle identifier from RAW**. `cycle_number` is
  taken exclusively from DOE.
- Capacity/energy are integrated as **magnitudes** using |I| and |I·V|, which
  is appropriate for most KPI ratios (e.g. CE, energy efficiency).
- CV/CC capacity integration is **contiguous‑run safe**: integration is done per
  contiguous `(step_type, cc_cv)` run and then summed, avoiding the classic
  over‑integration that happens when filtering discontiguous CV rows and
  trapezoiding across gaps.
"""

# ==============================================================================
# Low‑level helpers
# ==============================================================================

def _ensure_sorted(df: pd.DataFrame, by: str = "time_s") -> pd.DataFrame:
    """Return a frame sorted by `by` (stable), or the input if already monotonic.

    Parameters
    ----------
    df : pd.DataFrame
        Input frame.
    by : str, default "time_s"
        Column to use for ordering.

    Returns
    -------
    pd.DataFrame
        Sorted frame if required; otherwise the original reference.
    """
    if df.empty:
        return df
    col = df[by]
    if not col.is_monotonic_increasing:
        return df.sort_values(by, kind="mergesort")
    return df


def _integrate_trapz_abs_sorted(df: pd.DataFrame) -> Tuple[float, float, int, int]:
    """Integrate |I| and |I·V| over strictly positive time deltas.

    Preconditions
    -------------
    `df` must be sorted by `time_s` ascending and contain columns
    `time_s`, `current_a`, `voltage_v` without non‑numeric types.

    Returns
    -------
    (Q_Ah, E_Wh, n_rows_used, n_pairs_used)
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
    valid = dt > 0
    if not np.any(valid):
        return 0.0, 0.0, int(len(w)), 0

    I_mid = 0.5 * (I[:-1] + I[1:])
    IV_mid = 0.5 * (np.abs(I[:-1] * V[:-1]) + np.abs(I[1:] * V[1:]))

    Q_As = float(np.sum(I_mid[valid] * dt[valid]))
    E_Ws = float(np.sum(IV_mid[valid] * dt[valid]))
    return Q_As / 3600.0, E_Ws / 3600.0, int(len(w)), int(np.sum(valid))


def _sum_duration_by_mask_sorted(time_s: pd.Series, mask_sorted: pd.Series) -> float:
    """Sum duration in seconds where the boolean mask holds at **both** endpoints.

    Parameters
    ----------
    time_s : pd.Series
        Monotonic time stamps in seconds.
    mask_sorted : pd.Series
        Boolean mask aligned to `time_s` index.

    Returns
    -------
    float
        Total duration in seconds.
    """
    if len(time_s) < 2:
        return 0.0
    t = time_s.to_numpy()
    dt = np.diff(t)
    m = mask_sorted.to_numpy(dtype=bool)
    valid = (dt > 0) & m[:-1] & m[1:]
    return float(np.sum(dt[valid]))


# ==============================================================================
# Contiguous‑run aggregators
# ==============================================================================

def _sum_over_step_runs_sorted(df_sorted: pd.DataFrame, target_step: str) -> Tuple[float, float, Dict[str, int]]:
    """Integrate Q/E over contiguous runs where `step_type == target_step`.

    Parameters
    ----------
    df_sorted : pd.DataFrame
        RAW rows for a single cycle, pre‑sorted and annotated by `_prepare_for_basis`.
    target_step : str
        Step label, e.g. "charge" or "discharge" (case‑insensitive).

    Returns
    -------
    (Q_Ah, E_Wh, stats)
        `stats` includes counts of runs, rows, and positive‑dt pairs.
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
    """Sum device‑reported Ah over contiguous runs where `step_type == target_step`.

    Uses the counter column `step_charge_ah` and accumulates |last − first| per run
    to be resilient to non‑zero starting values.

    Returns
    -------
    (Q_meas_Ah, stats)
        If the column is absent, returns (NaN, zeros).
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
    """Integrate charge‑time CV and CC capacities per contiguous `(step_type, cc_cv)` run.

    Parameters
    ----------
    df_sorted : pd.DataFrame
        RAW rows for a single cycle, pre‑sorted and annotated by `_prepare_for_basis`.

    Returns
    -------
    (Q_CV_Ah, Q_CC_Ah, stats)
        Returns NaN for both capacities if no CV/CC runs are found. `stats` contains
        counts of runs, rows, and positive‑dt pairs per mode.
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
        Q, _, n_rows, n_pairs = _integrate_trapz_abs_sorted(sub)
        if ccv == "CV":
            Qcv += Q
            runs_cv += 1
            rows_cv += n_rows
            pairs_cv += n_pairs
        elif ccv == "CC":
            Qcc += Q
            runs_cc += 1
            rows_cc += n_rows
            pairs_cc += n_pairs
        # Unlabeled modes are ignored by design.

    stats = {
        "runs_cv": runs_cv, "rows_cv": rows_cv, "pairs_cv": pairs_cv,
        "runs_cc": runs_cc, "rows_cc": rows_cc, "pairs_cc": pairs_cc,
    }
    if runs_cv == 0 and runs_cc == 0:
        return float("nan"), float("nan"), stats
    return float(Qcv), float(Qcc), stats


# ==============================================================================
# Single‑cycle context and basis
# ==============================================================================

class SingleCycleKPI(Protocol):
    """Interface for a KPI unit that consumes a cycle context and returns values.

    Attributes
    ----------
    name : str
        Column name used in the KPI output.
    requires_basis : Tuple[str, ...]
        Keys that must exist in `CycleContext.cache` before `compute` is called.
    """

    name: str
    requires_basis: Tuple[str, ...]

    def compute(self, ctx: "CycleContext") -> Dict[str, Any]:
        """Compute KPI values for the provided cycle context."""
        ...


@dataclass
class CycleContext:
    """Container for per‑cycle data used by KPI computations.

    Parameters
    ----------
    raw : pd.DataFrame
        RAW rows for exactly one logical cycle (charge/OCV/discharge).
    doe : pd.DataFrame
        DOE rows for the same cycle; must include `cycle_number` and `step_number`.

    Attributes
    ----------
    ids : Dict[str, Any]
        Contains `cycle_number` and (optionally) `block_id` for traceability.
    cache : Dict[str, Any]
        Basis values and intermediate results shared across KPI units.
    """

    raw: pd.DataFrame
    doe: pd.DataFrame
    ids: Dict[str, Any] = field(default_factory=dict)
    cache: Dict[str, Any] = field(default_factory=dict)

    def get_id_tuple(self) -> Tuple[Any, Any]:
        """Return `(block_id, cycle_number)` for convenience."""
        return self.ids.get("block_id"), self.ids.get("cycle_number")


def _extract_ids(ctx: CycleContext) -> None:
    """Populate `ctx.ids` using **DOE only**.

    The module intentionally ignores any cycle identifier present in RAW. The
    provided DOE slice is expected to be filtered to a single block (if `block_id`
    exists) and a single `cycle_number`.
    """
    cyc_val = None
    blk_val = None

    if "cycle_number" in ctx.doe.columns:
        u = pd.unique(ctx.doe["cycle_number"].dropna())
        if len(u) >= 1:
            cyc_val = u[0]

    if "block_id" in ctx.doe.columns:
        u2 = pd.unique(ctx.doe["block_id"].dropna())
        if len(u2) >= 1:
            blk_val = u2[0]

    ctx.ids["cycle_number"] = cyc_val
    ctx.ids["block_id"] = blk_val


def _prepare_for_basis(raw: pd.DataFrame) -> pd.DataFrame:
    """Return a time‑sorted copy with helper columns for efficient grouping.

    Adds
    ----
    `_step_lc` : lower‑cased `step_type`
    `_cccv_uc` : upper‑cased `cc_cv` (empty string if missing)
    `_run_step`: contiguous run id by `step_type`
    `_run_both`: contiguous run id by (`step_type`, `cc_cv`)
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

    s["_run_step"] = (s["_step_lc"] != s["_step_lc"].shift(1)).cumsum()
    s["_run_both"] = ((s["_step_lc"] != s["_step_lc"].shift(1)) | (s["_cccv_uc"] != s["_cccv_uc"].shift(1))).cumsum()
    return s


def _compute_basis(ctx: CycleContext) -> None:
    """Compute and cache basis quantities for one cycle.

    Cached keys (selection)
    -----------------------
    Q_charge_Ah, E_charge_Wh, Q_discharge_Ah, E_discharge_Wh,
    Charge_total_duration_s, Discharge_total_duration_s,
    Charge_CV_duration_s, Q_charge_Ah_CV, Q_charge_Ah_CC,
    Q_charge_Ah_by_cccv, cccv_vs_total_mismatch_%, flags, debug_stats, etc.
    """
    s = _prepare_for_basis(ctx.raw)

    # Step‑type totals
    Qc, Ec, statsC = _sum_over_step_runs_sorted(s, "charge")
    Qd, Ed, statsD = _sum_over_step_runs_sorted(s, "discharge")

    Qc_meas, statsC_meas = _sum_measured_step_ah_over_step_runs_sorted(s, "charge")
    Qd_meas, statsD_meas = _sum_measured_step_ah_over_step_runs_sorted(s, "discharge")

    has_charge = statsC["runs"] > 0
    has_discharge = statsD["runs"] > 0
    is_partial_cycle = not (has_charge and has_discharge)

    debug_stats = {
        "charge": {**statsC, "meas_runs": statsC_meas["runs"], "meas_rows": statsC_meas["rows"]},
        "discharge": {**statsD, "meas_runs": statsD_meas["runs"], "meas_rows": statsD_meas["rows"]},
    }

    # Durations
    charge_total_dur_s = float("nan")
    discharge_total_dur_s = float("nan")
    if {"time_s", "step_type"}.issubset(s.columns):
        step_lc = s["_step_lc"]
        charge_total_dur_s = _sum_duration_by_mask_sorted(s["time_s"], step_lc.eq("charge"))
        discharge_total_dur_s = _sum_duration_by_mask_sorted(s["time_s"], step_lc.eq("discharge"))

    # CC/CV features
    has_cc_cv_col = "cc_cv" in ctx.raw.columns
    charge_cv_dur_s = float("nan")
    q_charge_cv = float("nan")
    q_charge_cc = float("nan")
    q_charge_by_cccv = float("nan")
    mismatch_pct = float("nan")

    if has_cc_cv_col:
        step_lc = s["_step_lc"]
        ccv_uc = s["_cccv_uc"]
        charge_cv_dur_s = _sum_duration_by_mask_sorted(s["time_s"], step_lc.eq("charge") & ccv_uc.eq("CV"))

        q_charge_cv, q_charge_cc, cc_stats = _sum_over_charge_cccv_runs_sorted(s)
        denom = 0.0
        if np.isfinite(q_charge_cv):
            denom += q_charge_cv
        if np.isfinite(q_charge_cc):
            denom += q_charge_cc
        q_charge_by_cccv = denom if denom > 0 else float("nan")

        if np.isfinite(Qc) and Qc > 0 and np.isfinite(q_charge_by_cccv):
            mismatch_pct = 100.0 * abs(q_charge_by_cccv - Qc) / Qc

        debug_stats["charge"].update({
            "cccv_runs_cv": cc_stats.get("runs_cv", 0),
            "cccv_runs_cc": cc_stats.get("runs_cc", 0),
            "cccv_pairs_cv": cc_stats.get("pairs_cv", 0),
            "cccv_pairs_cc": cc_stats.get("pairs_cc", 0),
        })

    ctx.cache.update({
        # Totals
        "Q_charge_Ah": Qc,
        "E_charge_Wh": Ec,
        "Q_discharge_Ah": Qd,
        "E_discharge_Wh": Ed,
        "Q_charge_Ah_meas": Qc_meas,
        "Q_discharge_Ah_meas": Qd_meas,
        # Flags
        "has_charge": has_charge,
        "has_discharge": has_discharge,
        "is_partial_cycle": is_partial_cycle,
        # Diagnostics
        "debug_stats": debug_stats,
        # Durations and CC/CV split
        "Charge_total_duration_s": charge_total_dur_s,
        "Discharge_total_duration_s": discharge_total_dur_s,
        "Charge_CV_duration_s": charge_cv_dur_s,
        "Q_charge_Ah_CV": q_charge_cv,
        "Q_charge_Ah_CC": q_charge_cc,
        "Q_charge_Ah_by_cccv": q_charge_by_cccv,
        "cccv_vs_total_mismatch_%": mismatch_pct,
        "has_cc_cv_column": has_cc_cv_col,
    })


# ==============================================================================
# KPI units
# ==============================================================================

@dataclass
class QChargeAh:
    """Total charge capacity (Ah) integrated from RAW charge segments."""

    name: str = "Q_charge_Ah"
    requires_basis: Tuple[str, ...] = ("Q_charge_Ah",)

    def compute(self, ctx: CycleContext) -> Dict[str, Any]:
        return {self.name: float(ctx.cache["Q_charge_Ah"])}


@dataclass
class QDischargeAh:
    """Total discharge capacity (Ah) integrated from RAW discharge segments."""

    name: str = "Q_discharge_Ah"
    requires_basis: Tuple[str, ...] = ("Q_discharge_Ah",)

    def compute(self, ctx: CycleContext) -> Dict[str, Any]:
        return {self.name: float(ctx.cache["Q_discharge_Ah"])}


@dataclass
class EChargeWh:
    """Total charge energy (Wh) integrated as |I·V| over charge segments."""

    name: str = "E_charge_Wh"
    requires_basis: Tuple[str, ...] = ("E_charge_Wh",)

    def compute(self, ctx: CycleContext) -> Dict[str, Any]:
        return {self.name: float(ctx.cache["E_charge_Wh"])}


@dataclass
class EDischargeWh:
    """Total discharge energy (Wh) integrated as |I·V| over discharge segments."""

    name: str = "E_discharge_Wh"
    requires_basis: Tuple[str, ...] = ("E_discharge_Wh",)

    def compute(self, ctx: CycleContext) -> Dict[str, Any]:
        return {self.name: float(ctx.cache["E_discharge_Wh"])}


@dataclass
class CoulombicEfficiency:
    """Coulombic efficiency (fraction): Q_discharge_Ah / Q_charge_Ah."""

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
    """Coulombic efficiency using device counters (if available)."""

    name: str = "CoulombicEfficiency_meas"
    requires_basis: Tuple[str, ...] = ("Q_charge_Ah_meas", "Q_discharge_Ah_meas")

    def compute(self, ctx: CycleContext) -> Dict[str, Any]:
        Qc = ctx.cache.get("Q_charge_Ah_meas", float("nan"))
        Qd = ctx.cache.get("Q_discharge_Ah_meas", float("nan"))
        if np.isfinite(Qc) and np.isfinite(Qd) and Qc > 0:
            return {self.name: float(Qd / Qc)}
        return {self.name: float("nan")}


@dataclass
class ChargeCVDurationSeconds:
    """Total time (s) spent under CV during the charge step."""

    name: str = "Charge_CV_duration_s"
    requires_basis: Tuple[str, ...] = ("Charge_CV_duration_s",)

    def compute(self, ctx: CycleContext) -> Dict[str, Any]:
        return {self.name: float(ctx.cache["Charge_CV_duration_s"])}


@dataclass
class ChargeTotalDurationSeconds:
    """Total time (s) of the charge step (all modes)."""

    name: str = "Charge_total_duration_s"
    requires_basis: Tuple[str, ...] = ("Charge_total_duration_s",)

    def compute(self, ctx: CycleContext) -> Dict[str, Any]:
        return {self.name: float(ctx.cache["Charge_total_duration_s"])}


@dataclass
class ChargeCVTimeRatio:
    """Fraction of charge time spent under CV: Charge_CV_duration_s / Charge_total_duration_s."""

    name: str = "Charge_CV_time_ratio"
    requires_basis: Tuple[str, ...] = ("Charge_CV_duration_s", "Charge_total_duration_s")

    def compute(self, ctx: CycleContext) -> Dict[str, Any]:
        cv = ctx.cache["Charge_CV_duration_s"]
        total = ctx.cache["Charge_total_duration_s"]
        val = float(cv / total) if (isinstance(total, (int, float)) and total > 0) else float("nan")
        return {self.name: val}


@dataclass
class ChargeCVCapacityRatio:
    """Capacity‑based CV fraction: Q_charge_Ah_CV / (Q_charge_Ah_CV + Q_charge_Ah_CC).

    Falls back to `Q_charge_Ah_CV / Q_charge_Ah` when the split denominator is
    unavailable (e.g. missing `cc_cv` labels).
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
    """Diagnostic percent mismatch between split sum and total charge: |(CV+CC)−total|/total·100."""

    name: str = "cccv_vs_total_mismatch_%"
    requires_basis: Tuple[str, ...] = ("cccv_vs_total_mismatch_%",)

    def compute(self, ctx: CycleContext) -> Dict[str, Any]:
        return {self.name: float(ctx.cache["cccv_vs_total_mismatch_%"])}


@dataclass
class EnergyEfficiency:
    """Energy efficiency (fraction): E_discharge_Wh / E_charge_Wh."""

    name: str = "EnergyEfficiency"
    requires_basis: Tuple[str, ...] = ("E_charge_Wh", "E_discharge_Wh")

    def compute(self, ctx: CycleContext) -> Dict[str, Any]:
        ec = ctx.cache["E_charge_Wh"]
        ed = ctx.cache["E_discharge_Wh"]
        val = float(ed / ec) if (isinstance(ec, (int, float)) and ec > 0) else float("nan")
        return {self.name: val}


@dataclass
class VChargeAvg:
    """Average charge voltage (V): E_charge_Wh / Q_charge_Ah."""

    name: str = "V_charge_avg"
    requires_basis: Tuple[str, ...] = ("E_charge_Wh", "Q_charge_Ah")

    def compute(self, ctx: CycleContext) -> Dict[str, Any]:
        q = ctx.cache["Q_charge_Ah"]
        e = ctx.cache["E_charge_Wh"]
        val = float(e / q) if (isinstance(q, (int, float)) and q > 0) else float("nan")
        return {self.name: val}


@dataclass
class VDischargeAvg:
    """Average discharge voltage (V): E_discharge_Wh / Q_discharge_Ah."""

    name: str = "V_discharge_avg"
    requires_basis: Tuple[str, ...] = ("E_discharge_Wh", "Q_discharge_Ah")

    def compute(self, ctx: CycleContext) -> Dict[str, Any]:
        q = ctx.cache["Q_discharge_Ah"]
        e = ctx.cache["E_discharge_Wh"]
        val = float(e / q) if (isinstance(q, (int, float)) and q > 0) else float("nan")
        return {self.name: val}


@dataclass
class QChargeAhCV:
    """Charge capacity (Ah) accumulated under CV during charge (contiguous‑run safe)."""

    name: str = "Q_charge_Ah_CV"
    requires_basis: Tuple[str, ...] = ("Q_charge_Ah_CV",)

    def compute(self, ctx: CycleContext) -> Dict[str, Any]:
        return {self.name: float(ctx.cache["Q_charge_Ah_CV"])}


@dataclass
class QChargeAhCC:
    """Charge capacity (Ah) accumulated under CC during charge (contiguous‑run safe)."""

    name: str = "Q_charge_Ah_CC"
    requires_basis: Tuple[str, ...] = ("Q_charge_Ah_CC",)

    def compute(self, ctx: CycleContext) -> Dict[str, Any]:
        return {self.name: float(ctx.cache["Q_charge_Ah_CC"])}


@dataclass
class QChargeAhByCCCV:
    """Sum of CV and CC charge capacities (Ah) used as a split‑consistent denominator."""

    name: str = "Q_charge_Ah_by_cccv"
    requires_basis: Tuple[str, ...] = ("Q_charge_Ah_by_cccv",)

    def compute(self, ctx: CycleContext) -> Dict[str, Any]:
        return {self.name: float(ctx.cache["Q_charge_Ah_by_cccv"])}


# ==============================================================================
# Public API: single‑cycle and multi‑cycle drivers
# ==============================================================================

def compute_single_cycle_kpis(
    raw_cycle_df: pd.DataFrame,
    doe_cycle_df: pd.DataFrame,
    kpi_units: List[SingleCycleKPI],
) -> Dict[str, Any]:
    """Compute requested KPIs for one logical cycle.

    Parameters
    ----------
    raw_cycle_df : pd.DataFrame
        RAW rows corresponding to the DOE steps of a single `cycle_number`.
    doe_cycle_df : pd.DataFrame
        DOE rows for the same cycle; must contain `cycle_number`.
    kpi_units : list of `SingleCycleKPI`
        KPI units to evaluate for this cycle.

    Returns
    -------
    Dict[str, Any]
        Flat dict containing identifiers (`cycle_number`, optional `block_id`),
        requested KPI values, and flags/diagnostics.
    """
    ctx = CycleContext(raw=raw_cycle_df.copy(), doe=doe_cycle_df.copy())
    _extract_ids(ctx)
    _compute_basis(ctx)

    out: Dict[str, Any] = {
        "block_id": ctx.ids.get("block_id"),
        "cycle_number": ctx.ids.get("cycle_number"),
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


def compute_multi_cycle_kpi_table(
    raw_df: pd.DataFrame,
    doe_df: pd.DataFrame,
    kpi_units: List[SingleCycleKPI],
    cycles: Optional[List[Any]] = None,
    block_col: str = "block_id",
    step_col: str = "step_number",
    cycle_col: str = "cycle_number",
) -> pd.DataFrame:
    """Compute a KPI table across cycles enumerated from DOE.

    Parameters
    ----------
    raw_df : pd.DataFrame
        Full RAW table for the run.
    doe_df : pd.DataFrame
        DOE table containing at least `cycle_number` and `step_number`. It is
        assumed to be pre‑filtered to **one** `block_id` if that column exists.
    kpi_units : list of `SingleCycleKPI`
        KPI units to compute per cycle.
    cycles : list, optional
        Optional subset of cycle numbers to process. If None, all cycles in DOE.
    block_col : str, default "block_id"
        Column name for block identifier (if present) used for defensive filtering.
    step_col : str, default "step_number"
        Foreign key linking DOE to RAW.
    cycle_col : str, default "cycle_number"
        Cycle identifier column in DOE; this module uses DOE only.

    Returns
    -------
    pd.DataFrame
        One row per processed cycle with identifiers, KPIs, and flags.
    """
    if cycle_col not in doe_df.columns:
        raise KeyError(f"DOE is missing required cycle column '{cycle_col}'.")
    if step_col not in doe_df.columns or step_col not in raw_df.columns:
        raise KeyError(
            f"'step_number' mapping required. Missing in: "
            f"{'DOE ' if step_col not in doe_df.columns else ''}"
            f"{'RAW' if step_col not in raw_df.columns else ''}"
        )

    # Assert single block if present in DOE
    if block_col in doe_df.columns:
        u_blocks = pd.unique(doe_df[block_col].dropna())
        if len(u_blocks) > 1:
            raise ValueError(
                f"DOE contains multiple '{block_col}' values ({list(u_blocks)}). "
                "Provide a DOE slice for a single block."
            )
        block_value = u_blocks[0] if len(u_blocks) == 1 else None
    else:
        block_value = None

    all_cycles = pd.unique(doe_df[cycle_col].dropna())
    if cycles is not None:
        wanted = set(cycles)
        all_cycles = [c for c in all_cycles if c in wanted]

    rows: List[Dict[str, Any]] = []
    for c in all_cycles:
        doe_cycle = doe_df[doe_df[cycle_col] == c]
        step_ids = pd.unique(doe_cycle[step_col].dropna())
        if len(step_ids) == 0:
            continue

        raw_slice = raw_df[raw_df[step_col].isin(step_ids)].copy()
        if block_value is not None and block_col in raw_slice.columns:
            raw_slice = raw_slice[raw_slice[block_col] == block_value]

        res = compute_single_cycle_kpis(raw_slice, doe_cycle, kpi_units)
        # Ensure identifiers
        res["cycle_number"] = c
        if block_value is not None:
            res.setdefault("block_id", block_value)
        rows.append(res)

    return pd.DataFrame(rows)


# ==============================================================================
# Meta‑KPIs: retention utilities
# ==============================================================================

def add_capacity_retention(
    cycle_kpi_df: pd.DataFrame,
    discharge_col: str = "Q_discharge_Ah",
    block_col: str = "block_id",
    out_col: str = "CapacityRetention_%",
    cycle_col: str = "cycle_number",
) -> pd.DataFrame:
    """Add `CapacityRetention_%` per block (or globally) vs. the first valid cycle.

    Definition
    ----------
    CapacityRetention_% = 100 * Q_discharge_Ah / baseline, where baseline is the
    first cycle with positive `Q_discharge_Ah` within the same block (if present)
    after ordering by `cycle_number`.
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
    cycle_col: str = "cycle_number",
) -> pd.DataFrame:
    """Add `EnergyRetention_%` per block (or globally) vs. the first valid cycle.

    Definition
    ----------
    EnergyRetention_% = 100 * E_discharge_Wh / baseline, with baseline defined
    analogously to `add_capacity_retention` but using energy instead of capacity.
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
