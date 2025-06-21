# src/modules/tremor/utils.py

import logging
import pandas as pd
import numpy as np
from scipy import signal
from typing import Any, List

L = logging.getLogger(__name__)

# --- Filtering Functions ---

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    # Add a check to prevent filtering at or above the Nyquist frequency
    if cutoff >= nyq:
        L.warning(f"High-pass cutoff ({cutoff} Hz) is >= Nyquist frequency ({nyq} Hz). Skipping filter.")
        return None, None
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    # Add a check to prevent filtering at or above the Nyquist frequency
    if cutoff >= nyq:
        L.warning(f"Low-pass cutoff ({cutoff} Hz) is >= Nyquist frequency ({nyq} Hz). Skipping filter.")
        return None, None
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    # If the filter could not be created, return the original data
    if b is None or a is None:
        return data
    # Ensure data is not empty before filtering
    if data.size == 0:
        return data
    y = signal.filtfilt(b, a, data)
    return y

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    # If the filter could not be created, return the original data
    if b is None or a is None:
        return data
    # Ensure data is not empty before filtering
    if data.size == 0:
        return data
    y = signal.filtfilt(b, a, data)
    return y

def pre_low_pass_filter(data, columns, fs, high_cut=1):
    for col in columns:
        for coord in ['x', 'y', 'z']:
            if (col, coord) in data.columns:
                series = data.loc[:, pd.IndexSlice[col, coord]].astype(float).interpolate()
                filtered = butter_lowpass_filter(series.to_numpy(), high_cut, fs)
                data.loc[:, pd.IndexSlice[col, coord]] = filtered
    return data

def pre_high_pass_filter(data, columns, fs, low_cut=1):
    for col in columns:
        for coord in ['x', 'y', 'z']:
            if (col, coord) in data.columns:
                series = data.loc[:, pd.IndexSlice[col, coord]].astype(float).interpolate()
                filtered = butter_highpass_filter(series.to_numpy(), low_cut, fs)
                data.loc[:, pd.IndexSlice[col, coord]] = filtered
    return data

# --- General Helper Functions ---

def check_hand_(patient: Any) -> List[str]:
    """Determines which hand(s) have been tracked by checking for marker presence."""
    hands = []
    if any('left' in col[0] for col in patient.pose_estimation.columns if isinstance(col, tuple)):
        hands.append('left')
    if any('right' in col[0] for col in patient.pose_estimation.columns if isinstance(col, tuple)):
        hands.append('right')
    
    if not hands:
        L.warning(f"Could not determine active hands for {patient.patient_id}. Defaulting to both.")
        return ['left', 'right']
        
    return hands

def get_marker_columns(df: pd.DataFrame, marker_name: str) -> list:
    """Returns available coordinate columns for a given marker."""
    coords = []
    for axis in ['x', 'y', 'z']:
        col = (marker_name, axis)
        if col in df.columns:
            coords.append(col)
    return coords

def normalize_multiindex_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures columns are MultiIndex and normalizes names by removing 'marker_' prefix.
    """
    if not isinstance(df.columns, pd.MultiIndex):
        L.warning("DataFrame columns are not MultiIndex. Attempting conversion.")
        new_cols = []
        for col in df.columns:
            parts = str(col).rsplit('_', 1)
            if len(parts) == 2:
                new_cols.append(tuple(parts))
            else:
                new_cols.append((str(col), ''))
        df.columns = pd.MultiIndex.from_tuples(new_cols)

    new_cols = []
    for col in df.columns:
        first_level = str(col[0])
        if first_level.startswith("marker_"):
            new_first = first_level[len("marker_"):]
        else:
            new_first = first_level
        new_cols.append((new_first,) + col[1:])
    df.columns = pd.MultiIndex.from_tuples(new_cols)
    return df

def executive_summary(feature_df: pd.DataFrame) -> str:
    """Creates an executive summary from the detailed feature DataFrame."""
    summary_lines = []
    regions = ['proximal_arm', 'distal_arm', 'fingers']
    sides = ['left', 'right']
    
    patient_id = feature_df.index[0] if not feature_df.empty else "Unknown"
    summary_lines.append(f"Tremor Analysis Summary for: {patient_id}")
    summary_lines.append("-" * 30)
    
    found_features = False
    for region in regions:
        for side in sides:
            col_median_amp = f"pca_hilbert_median_amplitude_{region}_{side}"
            col_dom_freq = f"pca_power_spectral_dominant_frequency_{region}_{side}"
            col_inst_freq_std = f"pca_instantaneous_frequency_std_{region}_{side}"

            if col_median_amp in feature_df.columns and col_dom_freq in feature_df.columns:
                median_amp = feature_df.loc[patient_id, col_median_amp]
                dom_freq = feature_df.loc[patient_id, col_dom_freq]
                inst_freq_std = feature_df.loc[patient_id, col_inst_freq_std] if col_inst_freq_std in feature_df.columns else np.nan
                
                if pd.notna(median_amp) and pd.notna(dom_freq):
                    found_features = True
                    line = (
                        f"{region.replace('_', ' ').capitalize()} ({side.capitalize()}): "
                        f"Median Amplitude = {median_amp:.3f} (norm.), "
                        f"Dominant Frequency = {dom_freq:.2f} Hz"
                    )
                    if pd.notna(inst_freq_std):
                        line += f", Freq. Variability (Std) = {inst_freq_std:.2f} Hz"
                    summary_lines.append(line)
    
    if not found_features:
        return f"No tremor features could be calculated for {patient_id}."

    return "\n".join(summary_lines)