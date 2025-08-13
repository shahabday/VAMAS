import pandas as pd
import numpy as np

class Preprocessor:
    """
    Preprocessor for battery experiment data.
    - Segments OCV, charge, and discharge steps (currently for CC/CV protocols)
    - Computes a summary DOE/steps table
    - Adds C-rate estimation if cell capacity is known/assumed
    """

    def __init__(self, df: pd.DataFrame, cell_capacity_ah: float = 1.0,current_threshold=0.01):
        """
        Initialize with raw dataframe (after column mapping) and optional cell capacity.
        """
        self.df = df.copy()
        self.cell_capacity_ah = cell_capacity_ah or 1.0
        self.current_threshold = current_threshold
        self.segmented = None
        self.steps_df = None
        self.steps_df_clean = None

        self.metadata = {}

    def detect_steps(self, classify_fn=None):
        """
        Label each row with step_type and step_number.
        """
        if classify_fn is None:
            classify_fn = self._default_classify_row
        self.df['step_type'] = self.df.apply(classify_fn, axis=1)
        step_change = self.df['step_type'] != self.df['step_type'].shift(1)
        self.df['step_number'] = step_change.cumsum()
        return self.df

    def _default_classify_row(self, row):
        if abs(row['current_a']) < self.current_threshold:
            return "OCV"
        elif row['current_a'] > 0:
            return "charge"
        elif row['current_a'] < 0:
            return "discharge"
        else:
            return "unknown"
    
    def segment_cc_cv_threshold(self, current_col='current_a', voltage_col='voltage_v', step_col='step_number'):
        """
        Segment CC/CV/OCV steps based on current/voltage thresholds, marking each row.
        """
        epsilon = 0.001  # voltage threshold (V)
        current_thr = self.current_threshold  # OCV threshold (A)
        df = self.df.copy()
        df['cc_cv'] = None

        for step, group in df.groupby(step_col):
            idx = group.index
            current = group[current_col]
            voltage = group[voltage_col]
            # 1. OCV
            if current.abs().mean() < current_thr:
                df.loc[idx, 'cc_cv'] = 'OCV'
            # 2. Charge (mean current > 0)
            elif current.mean() > 0:
                v_max = voltage.max()
                cv_start = idx[(voltage >= v_max - epsilon)][0]
                df.loc[idx[idx < cv_start], 'cc_cv'] = 'CC'
                df.loc[idx[idx >= cv_start], 'cc_cv'] = 'CV'
            # 3. Discharge (mean current < 0)
            elif current.mean() < 0:
                v_min = voltage.min()
                cv_start = idx[(voltage <= v_min + epsilon)][0]
                df.loc[idx[idx < cv_start], 'cc_cv'] = 'CC'
                df.loc[idx[idx >= cv_start], 'cc_cv'] = 'CV'
            else:
                df.loc[idx, 'cc_cv'] = 'Other'
        self.segmented = df
        return df

    def make_steps_summary(self, step_col='step_number', cc_cv_col='cc_cv', voltage_col='voltage_v', current_col='current_a'):
        """
        Generate a summary table of all steps (DOE table) from segmented data.
        
        Parameters
        ----------
        step_col : str, optional
            Column name representing the step number (default: 'step_number').
        cc_cv_col : str, optional
            Column name with CC/CV/OCV classification (default: 'cc_cv').
        voltage_col : str, optional
            Column name for voltage (default: 'voltage_v').
        current_col : str, optional
            Column name for current (default: 'current_a').

        Returns
        -------
        pandas.DataFrame
            Summary table (DOE table) with one row per step, including:
            - Basic timing info (`start_time`, `end_time`, `duration_s`)
            - Mean voltage/current for OCV steps
            - CC/CV parameters for charge and discharge steps
            - New field `cc_cv_mode` indicating:
                'CC'   → step contains only constant current region
                'CV'   → step contains only constant voltage region
                'CCCV' → step contains both regions
                None   → step not classified as charge or discharge
        """
        if self.segmented is None:
            raise ValueError("Run segment_cc_cv_threshold() first!")

        step_summaries = []
        for step_no, step_df in self.segmented.groupby(step_col):
            step_type = step_df["step_type"].iloc[0]  # Step type from detect_steps()
            start_time = step_df['time_s'].iloc[0]
            end_time = step_df['time_s'].iloc[-1]
            duration = end_time - start_time

            param_dict = {
                'step_number': step_no,
                'step_type': step_type,
                'start_time': start_time,
                'end_time': end_time,
                'duration_s': duration
            }

            # Default cc_cv_mode
            cc_cv_mode = None

            # OCV summary
            if step_type == 'OCV':
                param_dict['mean_voltage'] = step_df[voltage_col].mean()
                param_dict['mean_current'] = step_df[current_col].mean()

            # Charge/Discharge summaries
            elif step_type in ['charge', 'discharge']:
                cc_mask = step_df[cc_cv_col] == 'CC'
                cv_mask = step_df[cc_cv_col] == 'CV'
                prefix = step_type

                # Determine CC/CV composition
                if cc_mask.any() and not cv_mask.any():
                    cc_cv_mode = 'CC'
                elif cv_mask.any() and not cc_mask.any():
                    cc_cv_mode = 'CV'
                elif cc_mask.any() and cv_mask.any():
                    cc_cv_mode = 'CCCV'

                if cc_mask.any():
                    param_dict[f'{prefix}_cc_current'] = step_df.loc[cc_mask, current_col].mean()
                    param_dict[f'{prefix}_cc_duration'] = step_df.loc[cc_mask, 'time_s'].iloc[-1] - step_df.loc[cc_mask, 'time_s'].iloc[0]
                    param_dict[f'{prefix}_cc_cutoff_voltage'] = (
                        step_df.loc[cc_mask, voltage_col].max()
                        if prefix == 'charge' else step_df.loc[cc_mask, voltage_col].min()
                    )
                if cv_mask.any():
                    param_dict[f'{prefix}_cv_voltage'] = step_df.loc[cv_mask, voltage_col].mean()
                    param_dict[f'{prefix}_cv_duration'] = step_df.loc[cv_mask, 'time_s'].iloc[-1] - step_df.loc[cv_mask, 'time_s'].iloc[0]
                    param_dict[f'{prefix}_cv_cutoff_current'] = step_df.loc[cv_mask, current_col].abs().min()

            # Add CC/CV mode field
            param_dict['cc_cv_mode'] = cc_cv_mode

            step_summaries.append(param_dict)

        self.steps_df = pd.DataFrame(step_summaries)
        return self.steps_df


    def clean_steps(self, min_duration=0.5):
        """
        Remove steps shorter than min_duration (in seconds).
        """
        if self.steps_df is None:
            raise ValueError("Run make_steps_summary() first!")
        self.steps_df_clean = self.steps_df[self.steps_df['duration_s'] > min_duration].reset_index(drop=True)
        return self.steps_df_clean

    def estimate_c_rate(self):
        """
        Estimate C-rate for each step, using provided/assumed cell capacity.
        """
        if self.steps_df_clean is None:
            raise ValueError("Run clean_steps() first!")
        steps = self.steps_df_clean
        cap = self.cell_capacity_ah
        def get_c_rate(current):
            return current / cap if cap else None
        steps['c_rate'] = None
        for i, row in steps.iterrows():
            if row['step_type'] == 'charge' and not pd.isna(row.get('charge_cc_current', None)):
                steps.at[i, 'c_rate'] = get_c_rate(row['charge_cc_current'])
            elif row['step_type'] == 'discharge' and not pd.isna(row.get('discharge_cc_current', None)):
                steps.at[i, 'c_rate'] = get_c_rate(abs(row['discharge_cc_current']))
        self.steps_df_clean = steps
        return steps

    def process_all(self):
        """
        Run full pipeline: segment, summarize, clean, estimate c-rate.
        """
        self.detect_steps()
        self.segment_cc_cv_threshold()
        self.make_steps_summary()
        self.clean_steps()
        self.estimate_c_rate()
        return self.segmented, self.steps_df_clean

    

