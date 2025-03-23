import numpy as np
import pandas as pd
from typing import Any, Tuple, Optional, Union, List
import logging

logger = logging.getLogger(__name__)


def check_hand_(patient: Any) -> List[str]:
    """
    Determines which hand(s) have been tracked for a given patient based on their labels.
    
    Parameters:
        patient: An object with a 'label' attribute (expected to be a dictionary).
    
    Returns:
        A list of hand identifiers (e.g., ['left', 'right']).
    """
    conditions = patient.label

    if conditions:
        if 'hand' not in conditions.keys():
            hand = ['left', 'right']
        elif isinstance(conditions['hand'], list):
            hand = conditions['hand']
        else:
            hand = [conditions['hand']]
    else:
        hand = ['left', 'right']
    
    return hand

def normalize_multiindex_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures that DataFrame columns are in a MultiIndex format.
    If the columns are a single index (e.g. 'marker_index_finger_tip_left_x'),
    they are converted to a MultiIndex of the form: ('index_finger_tip_left', 'x').
    Also removes a leading "marker_" prefix if detected.
    """
    if not isinstance(df.columns, pd.MultiIndex):
        new_cols = []
        for col in df.columns:
            parts = col.rsplit("_", 1)
            if len(parts) == 2:
                new_cols.append(tuple(parts))
            else:
                new_cols.append((col, ""))
        df.columns = pd.MultiIndex.from_tuples(new_cols)
    
    new_cols = []
    prefix_found = False
    for col in df.columns:
        first_level = col[0]
        if first_level.startswith("marker_"):
            prefix_found = True
            new_first = first_level[len("marker_"):]
        else:
            new_first = first_level
        new_cols.append((new_first,) + col[1:])
    if prefix_found:
        df.columns = pd.MultiIndex.from_tuples(new_cols)
    return df

def get_marker_columns(df: pd.DataFrame, marker_name: str) -> list:
    """
    Returns available coordinate columns for a given marker.
    Checks for 'x', 'y', and optionally 'z', returning a list of tuple keys.
    """
    coords = []
    for axis in ['x', 'y', 'z']:
        col = (marker_name, axis)
        if col in df.columns:
            coords.append(col)
    return coords

def rle(inarray: Union[np.ndarray, list]) -> Optional[Tuple[int, int]]:
    """
    Run-length encoding tailored for detecting sequences (e.g., for active periods).
    
    This function returns the starting and ending frame indices of the longest sequence 
    of True values in the input array.
    
    Parameters:
        inarray: A list or numpy array containing boolean values.
        
    Returns:
        A tuple (frame_start, frame_end) corresponding to the largest run of True values,
        or None if the input array is empty or contains no True values.
    """
    ia = np.asarray(inarray)
    n = len(ia)
    if n == 0:
        return None
    else:
        y = ia[1:] != ia[:-1]
        i = np.append(np.where(y), n - 1)
        z = np.diff(np.append(-1, i))
        p = np.cumsum(np.append(0, z))[:-1]
        iax = ia[i]
        
        true_idx = np.where(iax == True)[0]
        if len(true_idx) == 0:
            return None
        max_idx = true_idx[z[true_idx].argmax()]
        frame_start = p[max_idx]
        
        if p.shape[0] == max_idx + 1:
            frame_end = n
        else:
            frame_end = p[max_idx + 1]
    
        return frame_start, frame_end

def compute_spectrogram(
    x: np.ndarray, 
    fs: float, 
    min_freq: float = 0, 
    max_freq: float = 100
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Computes a spectrogram by sliding a window over the signal.
    
    Parameters:
        x (np.ndarray): Input 1D signal.
        fs (float): Sampling rate.
        min_freq (float): Minimum frequency to include.
        max_freq (float): Maximum frequency to include.
    
    Returns:
        A tuple containing a list of amplitude arrays and a list of frequency arrays.
    """
    pad_width = int(fs / 2)
    x = np.pad(x, (pad_width, pad_width), mode='symmetric')   
    
    amplitudes = []
    frequencies = []    
    window_length = int(fs)
    
    for i in range(len(x) - window_length + 1): 
        xs = x[i:i + window_length]
        n_samples = len(xs)
        amplitude = 2 / n_samples * np.abs(np.fft.fft(xs))
        amplitude = amplitude[1:int(len(amplitude) / 2)]
        
        frequency = np.fft.fftfreq(n_samples) * n_samples / (1 / fs * len(xs))
        frequency = frequency[1:int(len(frequency) / 2)]
        
        freq_limits = (frequency >= min_freq) & (frequency <= max_freq)
        amplitude = amplitude[freq_limits]
        frequency = frequency[freq_limits]        
        
        amplitudes.append(amplitude)
        frequencies.append(frequency)
        
    return amplitudes, frequencies

def meanfilt(x: np.ndarray, k: int) -> np.ndarray:
    """
    Apply a length-k mean filter to a 1D array x.
    Boundaries are extended by repeating endpoints.
    
    Parameters:
        x (np.ndarray): 1D input array.
        k (int): Length of the mean filter (must be odd).
    
    Returns:
        A numpy array of the same length as x after applying the mean filter.
    """
    assert k % 2 == 1, "Mean filter length must be odd."
    assert x.ndim == 1, "Input must be one-dimensional."
    
    k2 = (k - 1) // 2
    y = np.zeros((len(x), k), dtype=x.dtype)
    y[:, k2] = x
    for i in range(k2):
        j = k2 - i
        y[j:, i] = x[:-j]
        y[:j, i] = x[0]
        y[:-j, -(i + 1)] = x[j:]
        y[-j:, -(i + 1)] = x[-1]
    return np.mean(y, axis=1)

def filter_peaks_and_troughs(
    peaks: np.ndarray, 
    troughs: np.ndarray, 
    signal: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filters peaks and troughs in a signal by ensuring that between consecutive peaks/troughs,
    only the highest peak and lowest trough are retained.
    
    Parameters:
        peaks (np.ndarray): Indices of detected peaks.
        troughs (np.ndarray): Indices of detected troughs.
        signal (np.ndarray): The original signal.
    
    Returns:
        A tuple (filtered_peaks, filtered_troughs) containing the unique, filtered indices.
    """
    filtered_peaks = []
    filtered_troughs = []
    i, j = 0, 0

    while i < len(peaks) and j < len(troughs):
        current_peak = peaks[i]
        current_trough = troughs[j]

        while i < len(peaks) - 1 and peaks[i + 1] < current_trough:
            if signal[peaks[i + 1]] > signal[current_peak]:
                current_peak = peaks[i + 1]
            i += 1

        while j < len(troughs) - 1 and troughs[j + 1] < current_peak:
            if signal[troughs[j + 1]] < signal[current_trough]:
                current_trough = troughs[j + 1]
            j += 1

        filtered_peaks.append(current_peak)
        filtered_troughs.append(current_trough)

        i = np.searchsorted(peaks, current_trough, side='right')
        j = np.searchsorted(troughs, current_peak, side='right')

    return np.unique(peaks), np.unique(filtered_troughs)

def ensure_peak_to_trough(
    peaks: List[int], 
    troughs: List[int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ensures that peaks and troughs alternate in the correct order.
    If the order is reversed, this function swaps them accordingly.
    
    Parameters:
        peaks (List[int]): List of peak indices.
        troughs (List[int]): List of trough indices.
    
    Returns:
        Two numpy arrays representing properly ordered peaks and troughs.
    """
    if peaks[0] < troughs[0]:
        reverse = False
        x = peaks.copy()
        y = troughs.copy()
    else:
        reverse = True
        x = troughs.copy()
        y = peaks.copy()
        
    for _ in range(100):
        for i in range(len(x) - 1):
            try:
                if x[i + 1] < y[i]:
                    x.pop(i + 1)
            except Exception:
                continue
            
        for i in range(len(y) - 1):
            try:
                if y[i + 1] < x[i + 1]:
                    y.pop(i + 1)
            except Exception:
                continue  
            
    min_idx = min(len(x), len(y))
    x = x[:min_idx]
    y = y[:min_idx]

    if reverse:
        return np.array(y), np.array(x)
    else:
        return np.array(x), np.array(y)


def load_tracking_csv(csv_path: str, video_path: str) -> Optional[pd.DataFrame]:
    """
    Attempts to load and validate the tracking CSV file.
    
    Parameters:
        csv_path (str): Path to the CSV file.
        video_path (str): Path to the video file (used for logging context).
    
    Returns:
        A valid DataFrame if successful, or None if the CSV is empty, invalid,
        or not in the expected MultiIndex format.
    """
    try:
        data = pd.read_csv(csv_path, header=[0, 1], index_col=0)
        if data.empty:
            logger.info("Tracking CSV exists for %s but is empty; will re-run tracking.", video_path)
            return None
        if not isinstance(data.columns, pd.MultiIndex):
            logger.error("Tracking CSV for %s is not in the expected MultiIndex format.", video_path)
            return None

        return data
    except Exception as e:
        logger.warning("Error reading tracking CSV for %s: %s; will re-run tracking.", video_path, e)
        return None

def identify_active_time_period(structural_features: pd.DataFrame, fs: float) -> dict:
    """
    Identifies active time periods for each hand in the structural features DataFrame.
    
    For each hand (left/right), the function examines the features (columns) that contain
    hand-related data. For each feature that does not contain NaN values, it computes a spectrogram,
    averages the spectrogram data, and uses run-length encoding (rle) to determine the start and end
    frames of the active period.
    
    Parameters:
        structural_features (pd.DataFrame): DataFrame with structural features.
        fs (float): Sampling rate.
    
    Returns:
        A dictionary with keys 'left' and 'right', where each key maps to a dictionary of feature
        names and their corresponding (frame_start, frame_end) active time periods.
    """
    time_periods = {'left': {}, 'right': {}}
    
    # Remove the 'time' column if present.
    if 'time' in structural_features.columns:
        structural_features = structural_features.drop('time', axis=1)
    
    def col_contains_hand(col, hand):
        if isinstance(col, tuple):
            return hand in col[0]
        return hand in col

    for hand in time_periods.keys():
        features = [col for col in structural_features.columns if col_contains_hand(col, hand)]
        for feat in features:
            if not structural_features.loc[:, feat].isna().any():
                amplitudes, _ = compute_spectrogram(structural_features.loc[:, feat].values, fs)
                spectrogram_data = np.vstack(amplitudes).T
                above_threshold = spectrogram_data.mean(axis=0) > np.median(spectrogram_data)
                rle_result = rle(above_threshold)
                if rle_result is not None:
                    frame_start, frame_end = rle_result
                    time_periods[hand][feat] = (frame_start, frame_end)
                
    return time_periods

def executive_summary(feature_df: pd.DataFrame) -> str:
    """
    Creates an executive summary from the detailed feature DataFrame.
    The summary highlights key tremor metrics for proximal, distal, and finger tremors on each side.
    
    Parameters:
        feature_df (pd.DataFrame): DataFrame containing detailed tremor features.
    
    Returns:
        A string summarizing the key tremor metrics.
    """
    summary_lines = []
    regions = ['proximal_arm', 'distal_arm', 'fingers']
    sides = ['left', 'right']
    
    for region in regions:
        for side in sides:
            col_max_amp = f"pca_hilbert_max_amplitude_{region}_{side}"
            col_median_amp = f"pca_hilbert_median_amplitude_{region}_{side}"
            col_amp_variance = f"pca_hilbert_amplitude_variance_{region}_{side}"
            
            col_median_freq = f"pca_power_spectral_median_frequency_{region}_{side}"
            col_dom_freq = f"pca_power_spectral_dominant_frequency_{region}_{side}"
            col_freq_variance = f"pca_power_spectral_frequency_variance_{region}_{side}"
            
            # New: instantaneous frequency variability (standard deviation).
            col_inst_freq_std = f"pca_instantaneous_frequency_std_{region}_{side}"
            
            # Check if the essential columns exist.
            if (col_max_amp in feature_df.columns and 
                col_median_amp in feature_df.columns and 
                col_median_freq in feature_df.columns):
                
                # Compute amplitude metrics.
                max_amp = feature_df[col_max_amp].mean()
                median_amp = feature_df[col_median_amp].mean()
                amp_var = (feature_df[col_amp_variance].mean() 
                           if col_amp_variance in feature_df.columns else None)
                
                # Compute frequency metrics.
                median_freq = feature_df[col_median_freq].mean()
                dom_freq = (feature_df[col_dom_freq].mean() 
                            if col_dom_freq in feature_df.columns else None)
                freq_var = (feature_df[col_freq_variance].mean() 
                            if col_freq_variance in feature_df.columns else None)
                
                # Compute instantaneous frequency variability if available.
                inst_freq_std = (feature_df[col_inst_freq_std].mean() 
                                 if col_inst_freq_std in feature_df.columns else None)
                
                # Build the summary line.
                line = f"{region.replace('_', ' ').capitalize()} ({side}): " \
                       f"Max Amplitude = {max_amp:.2f}, Median Amplitude = {median_amp:.2f}"
                if amp_var is not None:
                    line += f", Amplitude Variance = {amp_var:.2f}"
                
                line += f", Median Frequency = {median_freq:.2f} Hz"
                if dom_freq is not None:
                    line += f", Dominant Frequency = {dom_freq:.2f} Hz"
                    
                if inst_freq_std is not None:
                    line += f", Inst. Freq. Std = {inst_freq_std:.2f} Hz"
                
                summary_lines.append(line)
    
    # Optionally include the frame rate if available.
    if 'frame_rate' in feature_df.columns:
        fr = feature_df['frame_rate'].iloc[0]
        summary_lines.append(f"Frame rate: {fr:.2f} FPS")
    
    return "\n".join(summary_lines)
