import os 
import sys
from pathlib import Path


# Add src/ to sys.path so you can import batterydata
sys.path.append(os.path.abspath('./src'))


import numpy as np
import pandas as pd

from batterydata.pipeline.cycle_kpi_engine import TableBundle, CycleKPIEngine, default_registry, EXAMPLE_COLUMNS

def make_tiny_df():
    # One synthetic cycle: charge @ +1A for 1800s, OCV 600s, discharge @ -1A for 1800s.
    t_chg = np.linspace(0, 1800, 181)
    t_ocv = np.linspace(1800, 2400, 61)
    t_dch = np.linspace(2400, 4200, 181)

    seg = []
    # charge rows
    for t in t_chg:
        seg.append(dict(time_s=t, voltage_v=4.0, current_a=1.0, charge_ah=np.nan, energy_wh=np.nan,
                        cycle_index=1, step_type="charge", cc_cv="CC", step_number=1))
    # ocv rows
    for t in t_ocv:
        seg.append(dict(time_s=t, voltage_v=4.05, current_a=0.0, charge_ah=np.nan, energy_wh=np.nan,
                        cycle_index=1, step_type="ocv", cc_cv=None, step_number=2))
    # discharge rows
    for t in t_dch:
        seg.append(dict(time_s=t, voltage_v=3.6, current_a=-1.0, charge_ah=np.nan, energy_wh=np.nan,
                        cycle_index=1, step_type="discharge", cc_cv="CC", step_number=3))

    segmented = pd.DataFrame(seg)

    doe = pd.DataFrame([
        dict(step_number=1, step_type="charge", start_time=0, end_time=1800, duration_s=1800, cc_cv_mode="CC",
             cycle_number=1, step_sig="CH_1.00_V4.2_CV"),  # label OK even if we simulated only CC
        dict(step_number=2, step_type="ocv", start_time=1800, end_time=2400, duration_s=600, cc_cv_mode=None,
             cycle_number=1, step_sig="OCV_600"),
        dict(step_number=3, step_type="discharge", start_time=2400, end_time=4200, duration_s=1800, cc_cv_mode="CC",
             cycle_number=1, step_sig="DIS_1.00_V3.0_noCV"),
    ])

    meta = {"chemistry": "LCO", "nominal_capacity_ah": 2.0}
    columns = EXAMPLE_COLUMNS
    return segmented, doe, meta, columns

def test_scalar_kpis_basic():
    segmented, doe, meta, columns = make_tiny_df()
    tables = TableBundle(segmented, doe, meta, columns)
    engine = CycleKPIEngine(tables, default_registry())
    row = engine.compute_cycle(1)

    # Capacity should be ~0.5Ah (1A * 1800s / 3600)
    assert abs(row['Q_chg_Ah'] - 0.5) < 1e-3
    assert abs(row['Q_dch_Ah'] - 0.5) < 1e-3
    assert abs(row['CE_%'] - 100.0) < 1e-6

    # Durations from DOE
    assert row['t_CC_s'] >= 1800  # includes charge CC only
    assert row['t_dch_s'] == 1800

    # OCV slope ~0 since voltage constant
    assert abs(row['OCV_dVdt']) < 1e-9 or np.isnan(row['OCV_dVdt']) == False
    