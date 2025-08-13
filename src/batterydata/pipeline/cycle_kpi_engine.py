"""
Cycle KPI Engine — single-cell analysis

Design goals:
- Ontology-driven (column names provided via a mapping dict)
- Modular metrics (Strategy pattern + registry)
- Works per cycle, with optional block support later
- Complements existing DOE/segmented tables

Dependencies: pandas, numpy
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, Iterable
import numpy as np
import pandas as pd

# -----------------------------
# Column ontology (user-provided)
# -----------------------------

ColumnsMap = Dict[str, Dict[str, str]]


# -----------------------------
# Core data containers
# -----------------------------

@dataclass
class TableBundle:
    segmented_data: pd.DataFrame
    doe_table: pd.DataFrame
    metadata: Dict[str, Any]
    columns: ColumnsMap

    def __post_init__(self):
        # Basic validation: ensure required logical keys exist in mapping
        required_seg = [
            "time", "voltage", "current", "charge_ah", "energy_wh",
            "cycle", "step_type", "cc_cv_mode", "step_number"
        ]
        required_doe = [
            "step_number", "step_type", "start", "end", "duration_s",
            "cc_cv_mode", "cycle", "step_sig"
        ]
        for k in required_seg:
            if k not in self.columns.get("segmented", {}):
                raise ValueError(f"Missing segmented column mapping for '{k}'")
        for k in required_doe:
            if k not in self.columns.get("doe", {}):
                raise ValueError(f"Missing DOE column mapping for '{k}'")

        # Ensure required columns exist in dataframes
        seg_map = self.columns["segmented"]
        doe_map = self.columns["doe"]
        missing_seg = [seg_map[k] for k in required_seg if seg_map[k] not in self.segmented_data.columns]
        missing_doe = [doe_map[k] for k in required_doe if doe_map[k] not in self.doe_table.columns]
        if missing_seg:
            raise ValueError(f"Segmented data missing columns: {missing_seg}")
        if missing_doe:
            raise ValueError(f"DOE table missing columns: {missing_doe}")


@dataclass(frozen=True)
class SelectionSpec:
    cycle: Optional[Union[int, Sequence[int]]] = None
    block_id: Optional[int] = None
    repeat_slice: Optional[Tuple[int, int]] = None  # start, stop (exclusive)
    pattern: Optional[str] = None  # Placeholder; future extension


@dataclass(frozen=True)
class CycleCurves:
    cycle_number: int
    charge_curve: pd.DataFrame
    discharge_curve: pd.DataFrame
    ocv_curves: List[pd.DataFrame]
    ocv_between_charge_and_discharge: Optional[pd.DataFrame]
    doe_slice: pd.DataFrame


class CurveBuilder:
    def __init__(self, tables: TableBundle):
        self.t = tables
        self.seg = tables.segmented_data
        self.doe = tables.doe_table
        self.Cs = tables.columns["segmented"]
        self.Cd = tables.columns["doe"]

    def build_cycle_curves(self, cycle_number: int) -> CycleCurves:
        Cd = self.Cd
        Cs = self.Cs

        doe_cyc = self.doe[self.doe[Cd["cycle"]] == cycle_number].sort_values(Cd["step_number"]).copy()
        if doe_cyc.empty:
            raise ValueError(f"No DOE rows for cycle {cycle_number}")

        # Identify step_numbers by type
        step_type_col = Cd["step_type"]
        stepnum_col = Cd["step_number"]

        ch_steps = doe_cyc[doe_cyc[step_type_col].str.lower() == "charge"][stepnum_col].tolist()
        dch_steps = doe_cyc[doe_cyc[step_type_col].str.lower() == "discharge"][stepnum_col].tolist()
        ocv_steps = doe_cyc[doe_cyc[step_type_col].str.lower() == "ocv"][stepnum_col].tolist()

        # Slice segmented rows by step_number sets
        seg_step_col = self.Cs["step_number"]
        charge_curve = self.seg[self.seg[seg_step_col].isin(ch_steps)].copy()
        discharge_curve = self.seg[self.seg[seg_step_col].isin(dch_steps)].copy()
        ocv_curves = []
        for sn in ocv_steps:
            ocv_curves.append(self.seg[self.seg[seg_step_col] == sn].copy())

        # Find the OCV between charge and discharge: first ocv step with step_number
        # strictly between last charge step and first discharge step
        ocv_between = None
        if ch_steps and dch_steps and ocv_steps:
            last_ch = max(ch_steps)
            first_dch = min(dch_steps)
            mid_ocv = [sn for sn in ocv_steps if last_ch < sn < first_dch]
            if mid_ocv:
                ocv_between = self.seg[self.seg[seg_step_col] == mid_ocv[0]].copy()

        return CycleCurves(
            cycle_number=cycle_number,
            charge_curve=charge_curve.sort_values(self.Cs["time"]),
            discharge_curve=discharge_curve.sort_values(self.Cs["time"]),
            ocv_curves=[df.sort_values(self.Cs["time"]) for df in ocv_curves],
            ocv_between_charge_and_discharge=None if ocv_between is None else ocv_between.sort_values(self.Cs["time"]),
            doe_slice=doe_cyc,
        )


class CycleContext:
    """Single-cell context for cross-cycle metrics and caches."""

    def __init__(self, tables: TableBundle, curve_builder: CurveBuilder):
        self.t = tables
        self.curve_builder = curve_builder
        self.Cs = tables.columns["segmented"]
        self.Cd = tables.columns["doe"]
        self._q_dch_cache: Dict[int, float] = {}
        self._e_dch_cache: Dict[int, float] = {}
        self._throughput_cache: Dict[int, Tuple[float, float]] = {}  # cycle -> (Ah_dch_cum, Wh_dch_cum)

        # Precompute discharge throughput cumulatives by cycle for fast lookup
        self._precompute_throughput()

    # ---------- helpers ----------
    def _integrate_current_ah(self, df: pd.DataFrame) -> float:
        if df.empty:
            return np.nan
        t = df[self.Cs["time"]].to_numpy()
        i = df[self.Cs["current"]].to_numpy()
        if len(t) < 2:
            return 0.0
        return float(np.trapz(i, t) / 3600.0)

    def _integrate_energy_wh(self, df: pd.DataFrame) -> float:
        if df.empty:
            return np.nan
        t = df[self.Cs["time"]].to_numpy()
        v = df[self.Cs["voltage"]].to_numpy()
        i = df[self.Cs["current"]].to_numpy()
        if len(t) < 2:
            return 0.0
        p = v * i
        return float(np.trapz(p, t) / 3600.0)

    def _precompute_throughput(self):
        Cd = self.Cd
        cycles = sorted(self.t.doe_table[Cd["cycle"]].dropna().unique())
        cum_Ah = 0.0
        cum_Wh = 0.0
        for cyc in cycles:
            curves = self.curve_builder.build_cycle_curves(int(cyc))
            # Discharge throughput only (common convention); change if needed
            q_dch = abs(self._integrate_current_ah(curves.discharge_curve))
            e_dch = abs(self._integrate_energy_wh(curves.discharge_curve))
            cum_Ah += 0.0 if np.isnan(q_dch) else q_dch
            cum_Wh += 0.0 if np.isnan(e_dch) else e_dch
            self._throughput_cache[int(cyc)] = (cum_Ah, cum_Wh)
            # Cache per-cycle too
            self._q_dch_cache[int(cyc)] = q_dch
            self._e_dch_cache[int(cyc)] = e_dch

    # ---------- public accessors ----------
    def discharge_capacity_at(self, cycle_number: int) -> float:
        return self._q_dch_cache.get(int(cycle_number), np.nan)

    def cumulative_discharge_throughput_at(self, cycle_number: int) -> Tuple[float, float]:
        """Return cumulative (Ah, Wh) up to and including this cycle."""
        return self._throughput_cache.get(int(cycle_number), (np.nan, np.nan))


# -----------------------------
# Metric calculators
# -----------------------------

class MetricResult:
    def __init__(self, name: str, level: str, value: Any, unit: Optional[str] = None, flags: Optional[List[str]] = None):
        self.name = name
        self.level = level  # 'scalar' | 'series' | 'flag'
        self.value = value
        self.unit = unit
        self.flags = flags or []


class MetricCalculator:
    name: str = ""
    unit: Optional[str] = None
    level: str = "scalar"  # default scalar
    requires: List[str] = []
    needs: List[str] = []  # e.g., ['charge', 'discharge', 'ocv', 'doe', 'context']

    def compute(self, curves: CycleCurves, ctx: CycleContext, columns: ColumnsMap, intermediate: Dict[str, Any]) -> MetricResult:
        raise NotImplementedError


# Helper: small integration utilities re-used by many metrics
class _Integrators:
    @staticmethod
    def q_ah(df: pd.DataFrame, Cs: Dict[str, str]) -> float:
        if df is None or df.empty:
            return np.nan
        t = df[Cs["time"]].to_numpy()
        i = df[Cs["current"]].to_numpy()
        if len(t) < 2:
            return 0.0
        return float(np.trapz(i, t) / 3600.0)

    @staticmethod
    def e_wh(df: pd.DataFrame, Cs: Dict[str, str]) -> float:
        if df is None or df.empty:
            return np.nan
        t = df[Cs["time"]].to_numpy()
        v = df[Cs["voltage"]].to_numpy()
        i = df[Cs["current"]].to_numpy()
        if len(t) < 2:
            return 0.0
        p = v * i
        return float(np.trapz(p, t) / 3600.0)


# --- Concrete metrics ---

class Qchg(MetricCalculator):
    name = "Q_chg_Ah"
    unit = "Ah"
    def compute(self, curves: CycleCurves, ctx: CycleContext, columns: ColumnsMap, intermediate: Dict[str, Any]) -> MetricResult:
        Cs = columns["segmented"]
        q = _Integrators.q_ah(curves.charge_curve, Cs)
        # Ensure positive value for charge capacity
        if not np.isnan(q):
            q = abs(q)
        intermediate[self.name] = q
        return MetricResult(self.name, "scalar", q, self.unit)


class Qdch(MetricCalculator):
    name = "Q_dch_Ah"
    unit = "Ah"
    def compute(self, curves: CycleCurves, ctx: CycleContext, columns: ColumnsMap, intermediate: Dict[str, Any]) -> MetricResult:
        Cs = columns["segmented"]
        q = _Integrators.q_ah(curves.discharge_curve, Cs)
        if not np.isnan(q):
            q = abs(q)
        intermediate[self.name] = q
        return MetricResult(self.name, "scalar", q, self.unit)


class CE(MetricCalculator):
    name = "CE_%"
    unit = "%"
    requires = ["Q_chg_Ah", "Q_dch_Ah"]
    def compute(self, curves: CycleCurves, ctx: CycleContext, columns: ColumnsMap, intermediate: Dict[str, Any]) -> MetricResult:
        q_chg = intermediate.get("Q_chg_Ah", np.nan)
        q_dch = intermediate.get("Q_dch_Ah", np.nan)
        val = np.nan
        if q_chg and not np.isnan(q_chg):
            val = 100.0 * (q_dch / q_chg) if q_dch is not None else np.nan
        return MetricResult(self.name, "scalar", float(val), self.unit)


class Echg(MetricCalculator):
    name = "E_chg_Wh"
    unit = "Wh"
    def compute(self, curves: CycleCurves, ctx: CycleContext, columns: ColumnsMap, intermediate: Dict[str, Any]) -> MetricResult:
        Cs = columns["segmented"]
        e = _Integrators.e_wh(curves.charge_curve, Cs)
        if not np.isnan(e):
            e = abs(e)
        intermediate[self.name] = e
        return MetricResult(self.name, "scalar", e, self.unit)


class Edch(MetricCalculator):
    name = "E_dch_Wh"
    unit = "Wh"
    def compute(self, curves: CycleCurves, ctx: CycleContext, columns: ColumnsMap, intermediate: Dict[str, Any]) -> MetricResult:
        Cs = columns["segmented"]
        e = _Integrators.e_wh(curves.discharge_curve, Cs)
        if not np.isnan(e):
            e = abs(e)
        intermediate[self.name] = e
        return MetricResult(self.name, "scalar", e, self.unit)


class EnergyEff(MetricCalculator):
    name = "Energy_eff_%"
    unit = "%"
    requires = ["E_chg_Wh", "E_dch_Wh"]
    def compute(self, curves: CycleCurves, ctx: CycleContext, columns: ColumnsMap, intermediate: Dict[str, Any]) -> MetricResult:
        e_chg = intermediate.get("E_chg_Wh", np.nan)
        e_dch = intermediate.get("E_dch_Wh", np.nan)
        val = np.nan
        if e_chg and not np.isnan(e_chg):
            val = 100.0 * (e_dch / e_chg) if e_dch is not None else np.nan
        return MetricResult(self.name, "scalar", float(val), self.unit)


class DurationsChargeCC(MetricCalculator):
    name = "t_CC_s"
    unit = "s"
    def compute(self, curves: CycleCurves, ctx: CycleContext, columns: ColumnsMap, intermediate: Dict[str, Any]) -> MetricResult:
        Cd = columns["doe"]
        df = curves.doe_slice
        mask = (df[Cd["step_type"]].str.lower() == "charge") & (df[Cd["cc_cv_mode"]].str.upper() == "CC")
        val = float(df.loc[mask, Cd["duration_s"]].sum()) if not df.empty else np.nan
        intermediate[self.name] = val
        return MetricResult(self.name, "scalar", val, self.unit)


class DurationsChargeCV(MetricCalculator):
    name = "t_CV_s"
    unit = "s"
    def compute(self, curves: CycleCurves, ctx: CycleContext, columns: ColumnsMap, intermediate: Dict[str, Any]) -> MetricResult:
        Cd = columns["doe"]
        df = curves.doe_slice
        mask = (df[Cd["step_type"]].str.lower() == "charge") & (df[Cd["cc_cv_mode"]].str.upper() == "CV")
        val = float(df.loc[mask, Cd["duration_s"]].sum()) if not df.empty else 0.0
        intermediate[self.name] = val
        return MetricResult(self.name, "scalar", val, self.unit)


class DurationsDischarge(MetricCalculator):
    name = "t_dch_s"
    unit = "s"
    def compute(self, curves: CycleCurves, ctx: CycleContext, columns: ColumnsMap, intermediate: Dict[str, Any]) -> MetricResult:
        Cd = columns["doe"]
        df = curves.doe_slice
        mask = (df[Cd["step_type"]].str.lower() == "discharge")
        val = float(df.loc[mask, Cd["duration_s"]].sum()) if not df.empty else np.nan
        intermediate[self.name] = val
        return MetricResult(self.name, "scalar", val, self.unit)


class DurationChargeTotal(MetricCalculator):
    name = "t_chg_s"
    unit = "s"
    requires = ["t_CC_s", "t_CV_s"]
    def compute(self, curves: CycleCurves, ctx: CycleContext, columns: ColumnsMap, intermediate: Dict[str, Any]) -> MetricResult:
        tcc = intermediate.get("t_CC_s", 0.0) or 0.0
        tcv = intermediate.get("t_CV_s", 0.0) or 0.0
        val = float(tcc + tcv)
        intermediate[self.name] = val
        return MetricResult(self.name, "scalar", val, self.unit)


class CVShare(MetricCalculator):
    name = "CV_share_%"
    unit = "%"
    requires = ["t_CC_s", "t_CV_s", "t_chg_s"]
    def compute(self, curves: CycleCurves, ctx: CycleContext, columns: ColumnsMap, intermediate: Dict[str, Any]) -> MetricResult:
        tcc = intermediate.get("t_CC_s", 0.0) or 0.0
        tcv = intermediate.get("t_CV_s", 0.0) or 0.0
        tchg = intermediate.get("t_chg_s", 0.0) or 0.0
        val = np.nan
        if tchg > 0:
            val = 100.0 * (tcv / tchg)
        return MetricResult(self.name, "scalar", float(val), self.unit)


class VavgDischarge(MetricCalculator):
    name = "Vavg_dch_V"
    unit = "V"
    requires = ["Q_dch_Ah", "E_dch_Wh"]
    def compute(self, curves: CycleCurves, ctx: CycleContext, columns: ColumnsMap, intermediate: Dict[str, Any]) -> MetricResult:
        q = intermediate.get("Q_dch_Ah", np.nan)
        e = intermediate.get("E_dch_Wh", np.nan)
        val = np.nan
        if q and not np.isnan(q) and q > 0:
            val = e / q
        return MetricResult(self.name, "scalar", float(val), self.unit)


class VmaxCharge(MetricCalculator):
    name = "Vmax_chg_V"
    unit = "V"
    def compute(self, curves: CycleCurves, ctx: CycleContext, columns: ColumnsMap, intermediate: Dict[str, Any]) -> MetricResult:
        Cs = columns["segmented"]
        val = float(curves.charge_curve[Cs["voltage"]].max()) if not curves.charge_curve.empty else np.nan
        return MetricResult(self.name, "scalar", val, self.unit)


class VminDischarge(MetricCalculator):
    name = "Vmin_dch_V"
    unit = "V"
    def compute(self, curves: CycleCurves, ctx: CycleContext, columns: ColumnsMap, intermediate: Dict[str, Any]) -> MetricResult:
        Cs = columns["segmented"]
        val = float(curves.discharge_curve[Cs["voltage"]].min()) if not curves.discharge_curve.empty else np.nan
        return MetricResult(self.name, "scalar", val, self.unit)


class Retention(MetricCalculator):
    name = "Capacity_retention_%"
    unit = "%"
    requires = ["Q_dch_Ah"]
    def compute(self, curves: CycleCurves, ctx: CycleContext, columns: ColumnsMap, intermediate: Dict[str, Any]) -> MetricResult:
        q_now = intermediate.get("Q_dch_Ah", np.nan)
        q_ref = ctx.discharge_capacity_at(1)
        val = np.nan
        if q_ref and not np.isnan(q_ref) and q_ref > 0:
            val = 100.0 * (q_now / q_ref)
        return MetricResult(self.name, "scalar", float(val), self.unit)


class Throughput(MetricCalculator):
    name = "Ah_throughput"
    unit = "Ah"
    def compute(self, curves: CycleCurves, ctx: CycleContext, columns: ColumnsMap, intermediate: Dict[str, Any]) -> MetricResult:
        ah, _ = ctx.cumulative_discharge_throughput_at(curves.cycle_number)
        return MetricResult(self.name, "scalar", float(ah), self.unit)


class ThroughputWh(MetricCalculator):
    name = "Wh_throughput"
    unit = "Wh"
    def compute(self, curves: CycleCurves, ctx: CycleContext, columns: ColumnsMap, intermediate: Dict[str, Any]) -> MetricResult:
        _, wh = ctx.cumulative_discharge_throughput_at(curves.cycle_number)
        return MetricResult(self.name, "scalar", float(wh), self.unit)


class OCVSlope(MetricCalculator):
    name = "OCV_dVdt"
    unit = "V/s"
    def compute(self, curves: CycleCurves, ctx: CycleContext, columns: ColumnsMap, intermediate: Dict[str, Any]) -> MetricResult:
        Cs = columns["segmented"]
        ocv = curves.ocv_between_charge_and_discharge
        if ocv is None or ocv.empty:
            return MetricResult(self.name, "scalar", np.nan, self.unit, flags=["ocv_between_missing"])
        t = ocv[Cs["time"]].to_numpy()
        v = ocv[Cs["voltage"]].to_numpy()
        if len(t) < 2:
            return MetricResult(self.name, "scalar", np.nan, self.unit, flags=["ocv_short"])
        dv = float(v[-1] - v[0])
        dt = float(t[-1] - t[0])
        val = dv / dt if dt else np.nan
        return MetricResult(self.name, "scalar", val, self.unit)


class QualityFlag(MetricCalculator):
    name = "quality_flag"
    unit = None
    requires = ["Q_chg_Ah", "Q_dch_Ah", "CE_%"]
    def compute(self, curves: CycleCurves, ctx: CycleContext, columns: ColumnsMap, intermediate: Dict[str, Any]) -> MetricResult:
        flags: List[str] = []
        q_chg = intermediate.get("Q_chg_Ah", np.nan)
        q_dch = intermediate.get("Q_dch_Ah", np.nan)
        ce = intermediate.get("CE_%", np.nan)
        if np.isnan(q_chg) or q_chg == 0:
            flags.append("missing_charge")
        if np.isnan(q_dch) or q_dch == 0:
            flags.append("missing_discharge")
        if not np.isnan(ce) and (ce < 95 or ce > 105):
            flags.append("ce_outlier")
        return MetricResult(self.name, "flag", ",".join(flags) if flags else "OK")


# -----------------------------
# KPI Registry & Engine
# -----------------------------

class KPIRegistry:
    def __init__(self):
        self._metrics: Dict[str, MetricCalculator] = {}

    def register(self, metric: MetricCalculator) -> "KPIRegistry":
        self._metrics[metric.name] = metric
        return self

    def get(self, names: Optional[Iterable[str]] = None) -> List[MetricCalculator]:
        if names is None:
            return list(self._metrics.values())
        return [self._metrics[n] for n in names if n in self._metrics]

    def resolve_order(self, names: Optional[Iterable[str]] = None) -> List[MetricCalculator]:
        # Topological sort based on 'requires'
        selected = self.get(names)
        name_to_metric = {m.name: m for m in selected}
        visited: Dict[str, int] = {}  # 0=unseen,1=visiting,2=done
        order: List[MetricCalculator] = []

        def dfs(m: MetricCalculator):
            state = visited.get(m.name, 0)
            if state == 1:
                raise RuntimeError(f"Cyclic dependency at metric '{m.name}'")
            if state == 2:
                return
            visited[m.name] = 1
            for dep in getattr(m, "requires", []) or []:
                if dep in name_to_metric:
                    dfs(name_to_metric[dep])
            visited[m.name] = 2
            order.append(m)

        for m in selected:
            dfs(m)
        return order


class CycleKPIEngine:
    def __init__(self, tables: TableBundle, registry: KPIRegistry):
        self.tables = tables
        self.builder = CurveBuilder(tables)
        self.ctx = CycleContext(tables, self.builder)
        self.registry = registry
        self.names = tables.columns.get("names", {})

    def compute_cycle(self, cycle_number: int, metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        curves = self.builder.build_cycle_curves(cycle_number)
        ordered = self.registry.resolve_order(metrics)
        intermediate: Dict[str, Any] = {}
        scalars: Dict[str, Any] = {"cycle_number": cycle_number}
        flags: List[str] = []

        for m in ordered:
            res = m.compute(curves, self.ctx, self.tables.columns, intermediate)
            # Map to canonical name if provided
            out_name = self.names.get(res.name, res.name)
            if res.level == "scalar":
                scalars[out_name] = res.value
            elif res.level == "flag":
                scalars[out_name] = res.value
                if res.flags:
                    flags.extend(res.flags)
            # series metrics could be added later

        # attach a combined flags column if any additional flags collected
        if flags:
            existing = scalars.get(self.names.get("quality_flag", "quality_flag"), "OK")
            if existing and existing != "OK":
                scalars[self.names.get("quality_flag", "quality_flag")] = f"{existing};{';'.join(flags)}"
            else:
                scalars[self.names.get("quality_flag", "quality_flag")] = ";".join(flags)
        return scalars

    def compute_cycles(self, cycle_numbers: Sequence[int], metrics: Optional[List[str]] = None) -> pd.DataFrame:
        rows = [self.compute_cycle(int(c), metrics=metrics) for c in cycle_numbers]
        return pd.DataFrame(rows).sort_values("cycle_number").reset_index(drop=True)


# -----------------------------
# Default registry factory
# -----------------------------

def default_registry() -> KPIRegistry:
    reg = KPIRegistry()
    reg.register(Qchg())
    reg.register(Qdch())
    reg.register(CE())
    reg.register(Echg())
    reg.register(Edch())
    reg.register(EnergyEff())
    reg.register(DurationsChargeCC())
    reg.register(DurationsChargeCV())
    reg.register(DurationChargeTotal())
    reg.register(DurationsDischarge())
    reg.register(CVShare())
    reg.register(VavgDischarge())
    reg.register(VmaxCharge())
    reg.register(VminDischarge())
    reg.register(Retention())
    reg.register(Throughput())
    reg.register(ThroughputWh())
    reg.register(OCVSlope())
    reg.register(QualityFlag())
    return reg


# -----------------------------
# Example columns mapping (adjust to your exact names)
# -----------------------------

EXAMPLE_COLUMNS: ColumnsMap = {
    "segmented": {
        "time": "time_s",
        "voltage": "voltage_v",
        "current": "current_a",
        "charge_ah": "charge_ah",
        "energy_wh": "energy_wh",
        "cycle": "cycle_index",
        "step_type": "step_type",
        "cc_cv_mode": "cc_cv",
        "step_number": "step_number",
        "block_id": "block_id",
    },
    "doe": {
        "step_number": "step_number",
        "step_type": "step_type",
        "start": "start_time",
        "end": "end_time",
        "duration_s": "duration_s",
        "cc_cv_mode": "cc_cv_mode",
        "cycle": "cycle_number",
        "step_sig": "step_sig",
        "block_id": "block_id",
    },
    "meta": {
        "chemistry": "chemistry",
        "nominal_capacity_ah": "nominal_capacity_ah",
    },
    "names": {
        # Canonical output names
        "Q_chg_Ah": "Q_chg_Ah",
        "Q_dch_Ah": "Q_dch_Ah",
        "CE_%": "CE_%",
        "E_chg_Wh": "E_chg_Wh",
        "E_dch_Wh": "E_dch_Wh",
        "Energy_eff_%": "Energy_eff_%",
        "t_CC_s": "t_CC_s",
        "t_CV_s": "t_CV_s",
        "t_chg_s": "t_chg_s",
        "t_dch_s": "t_dch_s",
        "CV_share_%": "CV_share_%",
        "Vavg_dch_V": "Vavg_dch_V",
        "Vmax_chg_V": "Vmax_chg_V",
        "Vmin_dch_V": "Vmin_dch_V",
        "Capacity_retention_%": "Capacity_retention_%",
        "Ah_throughput": "Ah_throughput",
        "Wh_throughput": "Wh_throughput",
        "OCV_dVdt": "OCV_dVdt",
        "quality_flag": "quality_flag",
    },
}
