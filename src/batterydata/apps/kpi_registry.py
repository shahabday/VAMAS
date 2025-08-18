"""
Registry of available KPI classes and preset groupings.

Centralizes KPI imports so UI/CLI can discover and present choices consistently.
"""

from __future__ import annotations
from typing import List, Dict, Type

# Import KPI classes from your pipeline
from batterydata.pipeline.kpi_engine import (
    QChargeAh, QDischargeAh, EChargeWh, EDischargeWh,
    CoulombicEfficiency, EnergyEfficiency, VChargeAvg, VDischargeAvg,
    ChargeTotalDurationSeconds, ChargeCVDurationSeconds,
    ChargeCVTimeRatio, ChargeCVCapacityRatio, CCVTotalMismatchPct,
    QChargeAhCV, QChargeAhCC, QChargeAhByCCCV,
)

_ALL: List[type] = [
    QChargeAh, QDischargeAh, EChargeWh, EDischargeWh,
    CoulombicEfficiency, EnergyEfficiency, VChargeAvg, VDischargeAvg,
    ChargeTotalDurationSeconds, ChargeCVDurationSeconds,
    ChargeCVTimeRatio, ChargeCVCapacityRatio, CCVTotalMismatchPct,
    QChargeAhCV, QChargeAhCC, QChargeAhByCCCV,
]

_PRESETS: Dict[str, List[type]] = {
    "BASIC": [
        QChargeAh, QDischargeAh, EChargeWh, EDischargeWh,
        VChargeAvg, VDischargeAvg,
    ],
    "EFFICIENCY": [
        CoulombicEfficiency, EnergyEfficiency,
    ],
    "CV_DIAGNOSTICS": [
        ChargeTotalDurationSeconds, ChargeCVDurationSeconds,
        ChargeCVTimeRatio, ChargeCVCapacityRatio, CCVTotalMismatchPct,
        QChargeAhCV, QChargeAhCC, QChargeAhByCCCV,
    ],
    "FULL": _ALL,
}


def list_all() -> List[type]:
    """All KPI classes available."""
    return list(_ALL)


def list_presets() -> List[str]:
    """Preset names."""
    return list(_PRESETS.keys())


def get_preset(name: str) -> List[type]:
    """Return KPI class list for a preset name; raises KeyError if not found."""
    return _PRESETS[name]


def resolve_kpis(preset: str | None = None, classes: List[type] | None = None) -> List[object]:
    """
    Instantiate KPI classes from either a preset or an explicit list of classes.

    Returns
    -------
    List[object]
        Instantiated KPI objects ready for compute_multi_cycle_kpi_table.
    """
    if preset:
        kls = get_preset(preset)
    else:
        kls = classes or []
    return [k() for k in kls]
