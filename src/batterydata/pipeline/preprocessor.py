"""
BatteryData: data preparation pipeline

Public API
----------
- Preprocessor: main class that turns raw (column-mapped) records into
  (segmented_df, doe_table) ready for relational databases.
- step_signature: builds a protocol signature string per step.
- detect_repeating_blocks_v2 / detect_repeating_blocks_with_steps: block finder (DO NOT MODIFY LOGIC).

Notes
-----
- Assumes input df has at least: [time_s, voltage_v, current_a].
- If available, also accepts: [temperature_c] (ignored for now).
- Designed to plug in after `batterydata.parsers.basytec_reader`.
"""
from __future__ import annotations
from typing import Callable, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

__all__ = [
    "Preprocessor",
    "step_signature",
    "detect_repeating_blocks_v2",
    "detect_repeating_blocks_with_steps",
]


# -----------------------------
# Signature utilities
# -----------------------------

def step_signature(row: pd.Series) -> str:
    """Create a compact textual signature for a step.

    The signature is intentionally human-readable so that protocol patterns are
    easy to reason about and visualize.
    """
    if row["step_type"] == "OCV":
        return f"OCV_{int(row['duration_s'])}"
    elif row["step_type"] == "charge":
        cv = f"{row.get('charge_cc_cutoff_voltage', np.nan):.1f}"
        cr = f"{row.get('c_rate', np.nan):.2f}"
        has_cv = "CV" if not pd.isna(row.get("charge_cv_duration")) else "noCV"
        return f"CH_{cr}_V{cv}_{has_cv}"
    elif row["step_type"] == "discharge":
        cv = f"{row.get('discharge_cc_cutoff_voltage', np.nan):.1f}"
        cr = f"{row.get('c_rate', np.nan):.2f}"
        has_cv = "CV" if not pd.isna(row.get("discharge_cv_duration")) else "noCV"
        return f"DIS_{cr}_V{cv}_{has_cv}"
    else:
        return str(row["step_type"])  # passthrough for unknowns


# -----------------------------
# Block detection (do not modify logic)
# -----------------------------

def detect_repeating_blocks_v2(
    seq: List[str],
    min_block_len: int = 2,
    max_block_len: int = 10,
    min_repeats: int = 2,
    max_mismatches: int = 0,
):
    i = 0
    n = len(seq)
    summary = []
    while i < n:
        best = None
        # Try all possible block lengths from longest to shortest
        for blen in range(max_block_len, min_block_len - 1, -1):
            if i + blen * min_repeats > n:
                continue
            block = seq[i : i + blen]
            reps = 1
            cur = i + blen
            while cur + blen <= n and sum([block[j] != seq[cur + j] for j in range(blen)]) <= max_mismatches:
                reps += 1
                cur += blen
            if reps >= min_repeats:
                best = (blen, reps, block)
                break  # break on longest found
        if best:
            blen, reps, block = best
            summary.append({"block": block, "count": reps, "start": i})
            i += reps * blen
        else:
            summary.append({"block": [seq[i]], "count": 1, "start": i})
            i += 1
    return summary


def detect_repeating_blocks_with_steps(
    seq: List[str],
    min_block_len: int = 2,
    max_block_len: int = 10,
    min_repeats: int = 2,
    max_mismatches: int = 0,
):
    i = 0
    n = len(seq)
    summary = []
    while i < n:
        best = None
        for blen in range(max_block_len, min_block_len - 1, -1):
            if i + blen * min_repeats > n:
                continue
            block = seq[i : i + blen]
            reps = 1
            cur = i + blen
            while (
                cur + blen <= n
                and sum([block[j] != seq[cur + j] for j in range(blen)]) <= max_mismatches
            ):
                reps += 1
                cur += blen
            if reps >= min_repeats:
                best = (blen, reps, block)
                break
        if best:
            blen, reps, block = best
            summary.append({"block": block, "count": reps, "start": i})
            i += reps * blen
        else:
            summary.append({"block": [seq[i]], "count": 1, "start": i})
            i += 1
    return summary


# -----------------------------
# Preprocessor
# -----------------------------

class Preprocessor:
    """
    Preprocessor for battery experiment data.
    - Segments OCV, charge, and discharge steps (currently for CC/CV protocols)
    - Computes a summary DOE/steps table
    - Adds C-rate estimation if cell capacity is known/assumed
    - Generates step signatures and detects repeating protocol blocks
    - Annotates DOE with block_id and cycle_number

    Outputs
    -------
    - self.segmented: row-wise segmentation dataframe
    - self.steps_df: raw step summary (DOE)
    - self.steps_df_clean: cleaned step summary
    - self.metadata: dict of run parameters suitable for a metadata table
    """

    def __init__(
        self,
        df: pd.DataFrame,
        cell_capacity_ah: float = 1.0,
        current_threshold: float = 0.01,
        min_V: float = 0.001,
    ):
        self.df = df.copy()
        self.cell_capacity_ah = cell_capacity_ah or 1.0
        self.current_threshold = current_threshold
        self.segmented: Optional[pd.DataFrame] = None
        self.steps_df: Optional[pd.DataFrame] = None
        self.steps_df_clean: Optional[pd.DataFrame] = None
        self.epsilon = min_V  # threshold to detect constant Voltage for CV phase

        self.metadata: Dict[str, object] = {}

    # -------------
    # Step labeling
    # -------------
    def detect_steps(self, classify_fn: Optional[Callable[[pd.Series], str]] = None) -> pd.DataFrame:
        if classify_fn is None:
            classify_fn = self._default_classify_row
        self.df["step_type"] = self.df.apply(classify_fn, axis=1)
        step_change = self.df["step_type"] != self.df["step_type"].shift(1)
        self.df["step_number"] = step_change.cumsum()
        return self.df

    def _default_classify_row(self, row: pd.Series) -> str:
        if abs(row["current_a"]) < self.current_threshold:
            return "OCV"
        elif row["current_a"] > 0:
            return "charge"
        elif row["current_a"] < 0:
            return "discharge"
        else:
            return "unknown"

    # ---------------------------
    # CC/CV segmentation
    # ---------------------------
    def segment_cc_cv_threshold(
        self,
        current_col: str = "current_a",
        voltage_col: str = "voltage_v",
        step_col: str = "step_number",
    ) -> pd.DataFrame:
        import pandas as pd

        epsilon = self.epsilon
        current_thr = self.current_threshold
        min_cv_duration_s = getattr(self, "min_cv_duration_s", 2.0)

        df = self.df.copy()
        df["cc_cv"] = pd.Series(index=df.index, dtype="object")

        def _enforce_min_cv_duration(step_sub: pd.DataFrame, provisional_cv_mask: pd.Series):
            if not provisional_cv_mask.any():
                return False, None

            first_cv_label = provisional_cv_mask[provisional_cv_mask].index[0]

            after = step_sub.loc[first_cv_label:]
            after_mask = provisional_cv_mask.loc[first_cv_label:]

            if (~after_mask).any():
                first_false_label = after_mask[~after_mask].index[0]
                cont_block = step_sub.loc[first_cv_label:first_false_label].iloc[:-1]
            else:
                cont_block = after

            if len(cont_block) <= 1:
                return False, first_cv_label

            duration = cont_block["time_s"].iloc[-1] - cont_block["time_s"].iloc[0]
            return duration >= min_cv_duration_s, first_cv_label

        for step, group in df.groupby(step_col, sort=False):
            idx = group.index
            current = group[current_col]
            voltage = group[voltage_col]

            if current.abs().mean() < current_thr:
                df.loc[idx, "cc_cv"] = "OCV"
                continue

            if current.mean() > 0:  # charge
                v_max = voltage.max()
                provisional_cv_mask = voltage >= (v_max - epsilon)
                ok, cv_start_label = _enforce_min_cv_duration(group, provisional_cv_mask)
                if ok:
                    df.loc[idx[idx < cv_start_label], "cc_cv"] = "CC"
                    df.loc[idx[idx >= cv_start_label], "cc_cv"] = "CV"
                else:
                    df.loc[idx, "cc_cv"] = "CC"
                continue

            if current.mean() < 0:  # discharge
                v_min = voltage.min()
                provisional_cv_mask = voltage <= (v_min + epsilon)
                ok, cv_start_label = _enforce_min_cv_duration(group, provisional_cv_mask)
                if ok:
                    df.loc[idx[idx < cv_start_label], "cc_cv"] = "CC"
                    df.loc[idx[idx >= cv_start_label], "cc_cv"] = "CV"
                else:
                    df.loc[idx, "cc_cv"] = "CC"
                continue

            df.loc[idx, "cc_cv"] = "Other"

        self.segmented = df
        return df

    # ---------------------------
    # DOE summary / steps table
    # ---------------------------
    def make_steps_summary(
        self,
        step_col: str = "step_number",
        cc_cv_col: str = "cc_cv",
        voltage_col: str = "voltage_v",
        current_col: str = "current_a",
    ) -> pd.DataFrame:
        if self.segmented is None:
            raise ValueError("Run segment_cc_cv_threshold() first!")

        step_summaries: List[Dict[str, object]] = []
        for step_no, step_df in self.segmented.groupby(step_col):
            step_type = step_df["step_type"].iloc[0]
            start_time = step_df["time_s"].iloc[0]
            end_time = step_df["time_s"].iloc[-1]
            duration = end_time - start_time

            param_dict: Dict[str, object] = {
                "step_number": step_no,
                "step_type": step_type,
                "start_time": start_time,
                "end_time": end_time,
                "duration_s": duration,
            }

            cc_cv_mode = None

            if step_type == "OCV":
                param_dict["mean_voltage"] = step_df[voltage_col].mean()
                param_dict["mean_current"] = step_df[current_col].mean()

            elif step_type in ["charge", "discharge"]:
                cc_mask = step_df[cc_cv_col] == "CC"
                cv_mask = step_df[cc_cv_col] == "CV"
                prefix = step_type

                if cc_mask.any() and not cv_mask.any():
                    cc_cv_mode = "CC"
                elif cv_mask.any() and not cc_mask.any():
                    cc_cv_mode = "CV"
                elif cc_mask.any() and cv_mask.any():
                    cc_cv_mode = "CCCV"

                if cc_mask.any():
                    param_dict[f"{prefix}_cc_current"] = step_df.loc[cc_mask, current_col].mean()
                    param_dict[f"{prefix}_cc_duration"] = step_df.loc[cc_mask, "time_s"].iloc[-1] - step_df.loc[cc_mask, "time_s"].iloc[0]
                    param_dict[f"{prefix}_cc_cutoff_voltage"] = (
                        step_df.loc[cc_mask, voltage_col].max()
                        if prefix == "charge"
                        else step_df.loc[cc_mask, voltage_col].min()
                    )
                if cv_mask.any():
                    param_dict[f"{prefix}_cv_voltage"] = step_df.loc[cv_mask, voltage_col].mean()
                    param_dict[f"{prefix}_cv_duration"] = step_df.loc[cv_mask, "time_s"].iloc[-1] - step_df.loc[cv_mask, "time_s"].iloc[0]
                    param_dict[f"{prefix}_cv_cutoff_current"] = step_df.loc[cv_mask, current_col].abs().min()

            param_dict["cc_cv_mode"] = cc_cv_mode

            step_summaries.append(param_dict)

        self.steps_df = pd.DataFrame(step_summaries)
        return self.steps_df

    # ---------------------------
    # Cleaning & C-rate
    # ---------------------------
    def clean_steps(self, min_duration: float = 0.5) -> pd.DataFrame:
        if self.steps_df is None:
            raise ValueError("Run make_steps_summary() first!")
        self.steps_df_clean = self.steps_df[self.steps_df["duration_s"] > min_duration].reset_index(drop=True)
        return self.steps_df_clean

    def estimate_c_rate(self) -> pd.DataFrame:
        if self.steps_df_clean is None:
            raise ValueError("Run clean_steps() first!")
        steps = self.steps_df_clean
        cap = self.cell_capacity_ah

        def get_c_rate(current):
            return current / cap if cap else None

        steps["c_rate"] = None
        for i, row in steps.iterrows():
            if row["step_type"] == "charge" and not pd.isna(row.get("charge_cc_current", None)):
                steps.at[i, "c_rate"] = get_c_rate(row["charge_cc_current"])
            elif row["step_type"] == "discharge" and not pd.isna(row.get("discharge_cc_current", None)):
                steps.at[i, "c_rate"] = get_c_rate(abs(row["discharge_cc_current"]))
        self.steps_df_clean = steps
        return steps

    # ---------------------------
    # Protocol discovery (blocks)
    # ---------------------------
    def build_protocol_signatures(self) -> List[str]:
        if self.steps_df_clean is None:
            raise ValueError("Run estimate_c_rate() first (or at least clean_steps())!")
        self.steps_df_clean = self.steps_df_clean.copy()
        self.steps_df_clean["step_sig"] = self.steps_df_clean.apply(step_signature, axis=1)
        return list(self.steps_df_clean["step_sig"].values)

    def find_repeating_blocks(
        self,
        min_block_len: int = 2,
        max_block_len: int = 10,
        min_repeats: int = 2,
        max_mismatches: int = 0,
        with_steps: bool = True,
    ) -> List[Dict[str, object]]:
        protocol = self.build_protocol_signatures()
        if with_steps:
            blocks = detect_repeating_blocks_with_steps(
                protocol,
                min_block_len=min_block_len,
                max_block_len=max_block_len,
                min_repeats=min_repeats,
                max_mismatches=max_mismatches,
            )
        else:
            blocks = detect_repeating_blocks_v2(
                protocol,
                min_block_len=min_block_len,
                max_block_len=max_block_len,
                min_repeats=min_repeats,
                max_mismatches=max_mismatches,
            )
        self.metadata["block_detection_params"] = {
            "min_block_len": min_block_len,
            "max_block_len": max_block_len,
            "min_repeats": min_repeats,
            "max_mismatches": max_mismatches,
        }
        self.metadata["block_summary"] = blocks
        return blocks

    def annotate_blocks(self, blocks: List[Dict[str, object]]) -> pd.DataFrame:
        if self.steps_df_clean is None:
            raise ValueError("Run estimate_c_rate() first (or at least clean_steps())!")
        steps = self.steps_df_clean.copy()
        steps["block_id"] = np.nan
        steps["cycle_number"] = np.nan

        for block_idx, block_info in enumerate(blocks):
            block = block_info["block"]
            count = block_info["count"]
            start = block_info["start"]
            block_len = len(block)
            for repeat in range(count):
                idx_start = start + repeat * block_len
                idx_end = idx_start + block_len
                steps.loc[idx_start:idx_end - 1, "block_id"] = block_idx
                steps.loc[idx_start:idx_end - 1, "cycle_number"] = repeat + 1  # 1-based

        steps["block_id"] = steps["block_id"].astype("Int64")
        steps["cycle_number"] = steps["cycle_number"].astype("Int64")
        self.steps_df_clean = steps
        return steps

    # ---------------------------
    # Orchestration
    # ---------------------------
    def process_all(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run full pipeline up to C-rate (no block detection).

        Returns
        -------
        (segmented_df, doe_table_clean)
        """
        self.detect_steps()
        self.segment_cc_cv_threshold()
        self.make_steps_summary()
        self.clean_steps()
        self.estimate_c_rate()
        return self.segmented, self.steps_df_clean

    def process_with_blocks(
        self,
        min_block_len: int = 4,
        max_block_len: int = 4,
        min_repeats: int = 2,
        max_mismatches: int = 0,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, object]]]:
        """Run full pipeline + block detection + annotation.

        Returns
        -------
        (segmented_df, doe_table_clean_annotated, blocks)
        """
        seg, doe = self.process_all()
        blocks = self.find_repeating_blocks(
            min_block_len=min_block_len,
            max_block_len=max_block_len,
            min_repeats=min_repeats,
            max_mismatches=max_mismatches,
            with_steps=True,
        )
        doe_annot = self.annotate_blocks(blocks)
        return seg, doe_annot, blocks

    # ---------------------------
    # Convenience / I/O glue for DB readiness
    # ---------------------------
    def export_tables(self) -> Dict[str, pd.DataFrame]:
        """Return a dict of dataframes suitable for writing to a relational DB.

        Tables
        ------
        - segmented_data
        - doe_table (cleaned and annotated if annotate_blocks() was run)
        - metadata (flattened key/value for run parameters)
        """
        if self.segmented is None or self.steps_df_clean is None:
            raise ValueError("Run at least process_all() before export_tables().")

        meta_records = []
        for k, v in self.metadata.items():
            meta_records.append({"key": k, "value": v})
        metadata_df = pd.DataFrame(meta_records)

        return {
            "segmented_data": self.segmented.reset_index(drop=True),
            "doe_table": self.steps_df_clean.reset_index(drop=True),
            "metadata": metadata_df,
        }


# -----------------------------
# Example usage (for docs/tests)
# -----------------------------
# from batterydata.parsers.basytec_reader import reader
# from batterydata.pipeline.preprocessor import Preprocessor
#
# raw_df = reader(path_or_buffer)
# pp = Preprocessor(raw_df, cell_capacity_ah=1.1)
# seg, doe, blocks = pp.process_with_blocks(min_block_len=4, max_block_len=4, min_repeats=2)
# tables = pp.export_tables()
# segmented_df = tables["segmented_data"]
# doe_table = tables["doe_table"]
# metadata_table = tables["metadata"]
