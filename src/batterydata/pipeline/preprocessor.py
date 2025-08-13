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

        Logic
        -----
        - OCV: mean(|I|) < self.current_threshold  → label entire step 'OCV'
        - Charge: mean(I) > 0; CV starts when V >= (V_max - epsilon)
        - Discharge: mean(I) < 0; CV starts when V <= (V_min + epsilon)
        - A provisional CV region must persist for at least `min_cv_duration_s`
        (contiguous in time) to be considered valid; otherwise the whole step is 'CC'.

        Notes
        -----
        - Requires 'step_number' and 'time_s' columns (created earlier in the pipeline).
        - Numeric NaNs propagate naturally.
        """
        import pandas as pd

        epsilon = 0.001  # voltage proximity (V) to declare CV
        current_thr = self.current_threshold  # OCV threshold (A)
        min_cv_duration_s = getattr(self, 'min_cv_duration_s', 2.0)  # default if not set on the class

        df = self.df.copy()
        df['cc_cv'] = pd.Series(index=df.index, dtype='object')

        def _enforce_min_cv_duration(step_sub, provisional_cv_mask):
            """
            Returns (ok_bool, first_cv_label_index) for the first contiguous CV block
            that starts at the first True in provisional_cv_mask and lasts at least
            min_cv_duration_s seconds. If not ok, returns (False, first_label_or_None).
            """
            if not provisional_cv_mask.any():
                return False, None

            first_cv_label = provisional_cv_mask[provisional_cv_mask].index[0]

            after = step_sub.loc[first_cv_label:]
            after_mask = provisional_cv_mask.loc[first_cv_label:]

            # Cut the contiguous True block starting at first_cv_label
            if (~after_mask).any():
                first_false_label = after_mask[~after_mask].index[0]
                cont_block = step_sub.loc[first_cv_label:first_false_label].iloc[:-1]  # exclude the False
            else:
                cont_block = after

            if len(cont_block) <= 1:
                return False, first_cv_label

            duration = cont_block['time_s'].iloc[-1] - cont_block['time_s'].iloc[0]
            return duration >= min_cv_duration_s, first_cv_label

        # Process each step independently
        for step, group in df.groupby(step_col, sort=False):
            idx = group.index
            current = group[current_col]
            voltage = group[voltage_col]

            # 1) OCV: mean |I| below threshold
            if current.abs().mean() < current_thr:
                df.loc[idx, 'cc_cv'] = 'OCV'
                continue

            # 2) Charge: mean I > 0  → CV near V_max
            if current.mean() > 0:
                v_max = voltage.max()
                provisional_cv_mask = (voltage >= (v_max - epsilon))
                ok, cv_start_label = _enforce_min_cv_duration(group, provisional_cv_mask)

                if ok:
                    df.loc[idx[idx < cv_start_label], 'cc_cv'] = 'CC'
                    df.loc[idx[idx >= cv_start_label], 'cc_cv'] = 'CV'
                else:
                    df.loc[idx, 'cc_cv'] = 'CC'
                continue

            # 3) Discharge: mean I < 0 → CV near V_min
            if current.mean() < 0:
                v_min = voltage.min()
                provisional_cv_mask = (voltage <= (v_min + epsilon))
                ok, cv_start_label = _enforce_min_cv_duration(group, provisional_cv_mask)

                if ok:
                    df.loc[idx[idx < cv_start_label], 'cc_cv'] = 'CC'
                    df.loc[idx[idx >= cv_start_label], 'cc_cv'] = 'CV'
                else:
                    df.loc[idx, 'cc_cv'] = 'CC'
                continue

            # Fallback
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

    

