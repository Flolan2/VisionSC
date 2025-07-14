# src/modules/tremor/patients.py

import logging
import numpy as np
import pandas as pd
from scipy import ndimage
from copy import deepcopy

# Import from the new utils.py file in the same directory
from .utils import pre_high_pass_filter, pre_low_pass_filter

MIN_TIME_SAMPLES = 100
L = logging.getLogger(__name__)

class PatientCollection:
    """A collection of patient objects."""
    def __init__(self):
        self.patients = []
        self._markers = None

    def add_patient_list(self, patient_list):
        for patient in patient_list:
            patient.id = len(self.patients)
            self.patients.append(patient)

    def get_patient_ids(self):
        return [patient.patient_id for patient in self.patients if not patient.disabled]

    @property
    def markers(self):
        if self._markers is None and self.patients:
            self._markers = self.patients[0].markers
        return self._markers
        
    def __len__(self):
        return sum([1 for patient in self.patients if not patient.disabled])

class Patient:
    """Class to encode a generic patient structure, handling data cleaning."""
    def __init__(self,
                 pose_estimation: pd.DataFrame,
                 sampling_frequency: float,
                 patient_id: str,
                 label=None,
                 low_cut=0.5,
                 high_cut=15,
                 clean=True,
                 normalize=True,
                 likelihood_cutoff=0.9,
                 scaling_factor=1.0,
                 spike_threshold=10,
                 interpolate_pose=True,
                 smooth='median',
                 smoothing_window_length=3):

        self.pose_estimation = pose_estimation.copy()
        self.sampling_frequency = sampling_frequency
        self.patient_id = patient_id
        self.label = label if label is not None else {}
        self.disabled = False
        self.id = None # Set by PatientCollection
        
        # Attributes to be populated by the pipeline
        self.intention_tremor_features: pd.DataFrame = pd.DataFrame()
        self.postural_tremor_features: pd.DataFrame = pd.DataFrame()
        self.proximal_tremor_features: pd.DataFrame = pd.DataFrame()
        self.distal_tremor_features: pd.DataFrame = pd.DataFrame()
        self.fingers_tremor_features: pd.DataFrame = pd.DataFrame()
        
        # Parameters for cleaning
        self.low_cut = low_cut
        self.high_cut = high_cut
        self.likelihood_cutoff = likelihood_cutoff
        self.scaling_factor = scaling_factor
        self.spike_threshold = spike_threshold
        self.interpolate_pose = interpolate_pose
        self.smooth = smooth
        self.window_length = smoothing_window_length
        
        self.cleaning_summary = {}

        if self.pose_estimation is not None:
            self._update_markers()
            self._check_length()
            self._check_time()

            if normalize:
                self.normalize_data()
            if clean:
                self.clean_data()
            
            self.cleaned = True

    def _check_length(self):
        if len(self.pose_estimation.index) <= MIN_TIME_SAMPLES:
            L.warning(f"Patient {self.patient_id} has too few samples ({len(self.pose_estimation.index)}). Disabling.")
            self.disabled = True

    def _update_markers(self):
        self.markers = list(set(self.pose_estimation.columns.droplevel(1).tolist()))
        if 'time' in self.markers:
            self.markers.remove('time')

    def _check_time(self):
        if 'time' not in self.pose_estimation.columns:
            self.pose_estimation['time'] = np.asarray(self.pose_estimation.index * (1 / self.sampling_frequency))

    def clean_data(self, apply_filters=True, apply_smoothing=True):
        """
        Cleaning and filtering data based on initialization parameters.
        Filtering and smoothing can be disabled for specific analyses.
        """
        L.info(f"--- Starting data cleaning for {self.patient_id} with cutoff {self.likelihood_cutoff:.2f} ---")
        
        self.pose_estimation, retention_percent = remove_low_confidence_data(
            self.pose_estimation, self.markers, self.likelihood_cutoff
        )
        self.cleaning_summary['data_retention_percent'] = retention_percent

        if self.interpolate_pose:
            L.debug("Interpolating missing data points.")
            self.pose_estimation = self.pose_estimation.interpolate(limit_direction='both', method='linear')

        # MODIFIED: Make filtering and smoothing optional
        if apply_filters:
            if self.spike_threshold > 0:
                L.debug(f"Removing spikes with threshold > {self.spike_threshold}.")
                self.pose_estimation = remove_spikes(self.pose_estimation, self.markers, threshold=self.spike_threshold)

            if self.low_cut > 0:
                L.debug(f"Applying high-pass filter with low_cut at {self.low_cut} Hz.")
                self.pose_estimation = filter_low_frequency(self.pose_estimation, self.markers, self.sampling_frequency, self.low_cut)

            if self.high_cut is not None:
                L.debug(f"Applying low-pass filter with high_cut at {self.high_cut} Hz.")
                self.pose_estimation = filter_high_frequency(self.pose_estimation, self.markers, self.sampling_frequency, self.high_cut)

        if apply_smoothing:
            if self.smooth == 'median':
                L.debug(f"Applying median smoothing with window length {self.window_length}.")
                self.pose_estimation = smooth_median(self.pose_estimation, self.markers, window_len=self.window_length)
        
        # Final interpolation to fill any remaining NaNs after cleaning
        self.pose_estimation.interpolate(limit_direction='both', inplace=True)
        L.info(f"--- Finished data cleaning for {self.patient_id} ---")


    def normalize_data(self):
        """Scales coordinate data. For our pipeline, this is a placeholder as z-scoring is done later."""
        if isinstance(self.scaling_factor, (float, int)) and self.scaling_factor != 1.0:
            marker_cols = [c for c in self.pose_estimation.columns if c[0] in self.markers and c[1] in ['x', 'y', 'z']]
            self.pose_estimation[marker_cols] *= self.scaling_factor
        else:
            L.debug("Using default scaling factor of 1.0 (no change).")

# --- Helper Functions for Data Cleaning (self-contained within this module) ---

def remove_spikes(data, markers, threshold=20):
    """Remove spikes for each marker."""
    for marker in markers:
        for coord in ['x', 'y', 'z']:
            if (marker, coord) in data.columns:
                data.loc[:, (marker, coord)] = _remove_single_spike_series(data.loc[:, (marker, coord)], threshold)
    return data

def _remove_single_spike_series(signal, threshold, median_window=5, std_window=21, threshold_factor=3):
    if median_window % 2 == 0: median_window += 1
    if std_window % 2 == 0: std_window += 1
    
    s = pd.Series(signal).copy()
    s_clean = s.interpolate(limit_direction='both') # Interpolate first to handle existing NaNs
    
    median_filtered = s_clean.rolling(window=median_window, center=True, min_periods=1).median()
    rolling_std = s_clean.rolling(window=std_window, center=True, min_periods=1).std()
    
    # Identify spikes where deviation from median is high relative to local std
    spikes = np.abs(s_clean - median_filtered) > (threshold_factor * rolling_std + 1e-6) # add epsilon to avoid division by zero
    
    s[spikes] = np.nan
    return s.to_numpy()

def smooth_median(data, markers, window_len=3):
    """Applies a median filter to each marker's time series."""
    data_smoothed = data.copy()
    for marker in markers:
        for coord in ['x', 'y', 'z']:
            if (marker, coord) in data_smoothed.columns:
                series_data = data_smoothed.loc[:, (marker, coord)].to_numpy()
                
                # Create a mask of NaN values
                mask = np.isnan(series_data)

                # Check if there are any valid data points to interpolate from
                if np.all(mask):
                    # If the entire series is NaN, there's nothing to do. Continue to the next one.
                    L.warning(f"Column ({marker}, {coord}) is all NaN. Skipping median smoothing for it.")
                    continue

                # Handle NaNs before filtering by interpolating only if needed
                if np.any(mask):
                    series_data[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), series_data[~mask])
                
                filtered_series = ndimage.median_filter(series_data, size=window_len)
                data_smoothed.loc[:, (marker, coord)] = filtered_series
    return data_smoothed


def remove_low_confidence_data(data, markers, confidence_cutoff):
    """
    Sets low-confidence samples to NaN and returns the cleaned data and retention percentage.
    It checks for 'visibility', 'presence', and 'likelihood' columns.
    """
    data_copy = data.copy()
    
    # Identify the name of the confidence metric used in the dataframe
    confidence_metric = None
    key_markers_to_check = ['right_shoulder', 'left_shoulder', 'right_wrist', 'left_wrist', 'nose']
    for metric in ['visibility', 'presence', 'likelihood']:
        if any((marker, metric) in data_copy.columns for marker in key_markers_to_check):
            confidence_metric = metric
            break
            
    if confidence_metric is None:
        L.warning("--> LOG: No confidence column found. Skipping confidence filtering.")
        # Calculate retention based on existing NaNs if any
        coord_cols = [c for c in data_copy.columns if c[1] in ['x', 'y', 'z']]
        points_before = data_copy[coord_cols].size
        points_after = data_copy[coord_cols].notna().sum().sum()
        retention = (points_after / points_before) * 100 if points_before > 0 else 100.0
        return data_copy, retention

    L.info(f"--> LOG: Using '{confidence_metric}' as the confidence metric with cutoff {confidence_cutoff:.2f}.")

    # --- CORRECTED LOGIC ---
    # Iterate through each marker that has a confidence score
    for marker in markers:
        confidence_col = (marker, confidence_metric)
        if confidence_col in data_copy.columns:
            # Create a boolean mask of all rows where confidence is LOW for this marker
            low_confidence_mask = data_copy[confidence_col] < confidence_cutoff
            
            # Find the coordinate columns for this specific marker
            marker_coord_cols = [
                (marker, 'x'),
                (marker, 'y'),
                (marker, 'z')
            ]
            
            # For each coordinate column that exists...
            for col in marker_coord_cols:
                if col in data_copy.columns:
                    # ...use the mask to set the low-confidence rows to NaN
                    data_copy.loc[low_confidence_mask, col] = np.nan

    # --- Calculate retention percentage AFTER modifications ---
    coord_cols = [c for c in data_copy.columns if c[1] in ['x', 'y', 'z']]
    points_before = data_copy[coord_cols].size
    points_after = data_copy[coord_cols].notna().sum().sum()
    retention_percent = (points_after / points_before) * 100 if points_before > 0 else 100.0
    
    L.info(f"--> LOG: Final data retention after cutoff {confidence_cutoff:.2f}: {retention_percent:.2f}%")

    # Drop all possible confidence-related columns after use
    cols_to_drop = [col for col in data_copy.columns if col[1] in ['likelihood', 'visibility', 'presence']]
    data_copy = data_copy.drop(columns=cols_to_drop, errors='ignore')
    
    return data_copy, retention_percent

def filter_low_frequency(data, markers, fps, low_cut=0):
    return pre_high_pass_filter(data, markers, fps, low_cut=low_cut)

def filter_high_frequency(data, markers, fps, high_cut=None):
    return pre_low_pass_filter(data, markers, fps, high_cut=high_cut)