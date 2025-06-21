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

    def clean_data(self):
        """Cleaning and filtering data based on initialization parameters."""
        if 'likelihood' in self.pose_estimation.columns.get_level_values(1):
            self.pose_estimation = remove_low_likelihood(self.pose_estimation, self.markers, self.likelihood_cutoff)

        if self.interpolate_pose:
            self.pose_estimation = self.pose_estimation.interpolate(limit_direction='both', method='linear')

        if self.spike_threshold > 0:
            self.pose_estimation = remove_spikes(self.pose_estimation, self.markers, threshold=self.spike_threshold)

        if self.low_cut > 0:
            self.pose_estimation = filter_low_frequency(self.pose_estimation, self.markers, self.sampling_frequency, self.low_cut)

        if self.high_cut is not None:
            self.pose_estimation = filter_high_frequency(self.pose_estimation, self.markers, self.sampling_frequency, self.high_cut)

        if self.smooth == 'median':
            self.pose_estimation = smooth_median(self.pose_estimation, self.markers, window_len=self.window_length)
        
        # Final interpolation to fill any remaining NaNs after cleaning
        self.pose_estimation.interpolate(limit_direction='both', inplace=True)


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


def remove_low_likelihood(data, markers, likelihood):
    """Setting low likelihood samples to NaN and dropping the likelihood column."""
    for marker in markers:
        if (marker, 'likelihood') in data.columns:
            low_conf_indices = data.loc[:, (marker, 'likelihood')] < likelihood
            for coord in ['x', 'y', 'z']:
                if (marker, coord) in data.columns:
                    data.loc[low_conf_indices, (marker, coord)] = np.nan
    
    likelihood_cols = [col for col in data.columns if col[1] in ['likelihood', 'visibility', 'presence']]
    data = data.drop(columns=likelihood_cols)
    return data

def filter_low_frequency(data, markers, fps, low_cut=0):
    return pre_high_pass_filter(data, markers, fps, low_cut=low_cut)

def filter_high_frequency(data, markers, fps, high_cut=None):
    return pre_low_pass_filter(data, markers, fps, high_cut=high_cut)