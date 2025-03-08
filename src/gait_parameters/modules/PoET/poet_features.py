#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
poet_feature_extraction.py
Created on [Date]
Author: [Your Name]

This module contains functions for normalizing multi-index DataFrame columns,
and extracting tremor features from structural data using PCA.
It leverages functions from the poet_signal_preprocessing module.
"""

import os
import logging
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Local utilities assumed to work with multi-index column keys.
from .poet_utils import identify_active_time_period, check_hand_, meanfilt

# Import signal processing functions from the separate module.
from .poet_signal_analysis import butter_bandpass_filter, pca_tremor_analysis

# Global output directory for plots.
PLOTS_DIR = "./plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

DEBUG = True
logger = logging.getLogger(__name__)

#############################################
# Utility Functions for Data Normalization
#############################################
def normalize_multiindex_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures that DataFrame columns are in a MultiIndex format.
    If the columns are a single index (e.g. 'marker_index_finger_tip_left_x'),
    they are converted to a MultiIndex of the form: ('marker_index_finger_tip_left', 'x').
    
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
    
    if DEBUG:
        logger.debug("First row of DataFrame:\n%s", df.head(1))
        logger.debug("Original column names: %s", df.columns.tolist())
    
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
        if DEBUG:
            logger.debug("Detected 'marker_' prefix. Removing prefix for consistency.")
        df.columns = pd.MultiIndex.from_tuples(new_cols)
    else:
        if DEBUG:
            logger.debug("No 'marker_' prefix detected. Using current column names.")
    if DEBUG:
        logger.debug("Normalized column names: %s", df.columns.tolist())
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

#############################################
# Active Time Period Assignment
#############################################
def assign_hand_time_periods(pc):
    """
    Loop over patients and assign active time periods.
    Uses an available tremor features DataFrame from the new dedicated attributes
    (proximal, distal, or fingers) or falls back to structural_features if available.
    Normalizes the DataFrame to ensure MultiIndex format and then computes
    active time periods.
    """
    logger.info("Extracting active time periods of hands ...")
    for p in tqdm(pc.patients, total=len(pc.patients)):
        fs = p.sampling_frequency
        
        # Choose an available structural data source from new attributes:
        structural_data = None
        if hasattr(p, 'proximal_tremor_features') and p.proximal_tremor_features is not None and not p.proximal_tremor_features.empty:
            structural_data = p.proximal_tremor_features
        elif hasattr(p, 'distal_tremor_features') and p.distal_tremor_features is not None and not p.distal_tremor_features.empty:
            structural_data = p.distal_tremor_features
        elif hasattr(p, 'fingers_tremor_features') and p.fingers_tremor_features is not None and not p.fingers_tremor_features.empty:
            structural_data = p.fingers_tremor_features
        elif hasattr(p, 'structural_features') and p.structural_features is not None and not p.structural_features.empty:
            structural_data = p.structural_features
        
        if structural_data is not None:
            structural_data = normalize_multiindex_columns(structural_data)
            if DEBUG:
                logger.debug("Patient %s structural features loaded. Columns: %s", 
                             p.patient_id, structural_data.columns.tolist())
                logger.debug("First two rows for patient %s:\n%s", p.patient_id, structural_data.head(2))
            hands = check_hand_(p)
            if len(hands) > 1:
                time_periods = identify_active_time_period(structural_data, fs)
            else:
                time_periods = None
            p.hand_time_periods = time_periods
        else:
            logger.debug("Patient %s has no structural features available for time period extraction.", p.patient_id)
            p.hand_time_periods = None
    return pc

#############################################
# Helper for Marker-Pair Feature Extraction
#############################################
def _extract_tremor_features_for_marker_pair(p, fs, structural_features: pd.DataFrame,
                                             marker_a: str, marker_b: str,
                                             hand: str, time_periods: dict,
                                             debug: bool, plot: bool, save_plots: bool,
                                             feature_prefix: str) -> dict:
    """
    Extract tremor features for a given patient and a pair of markers.
    This helper function:
      - Determines the proper time window,
      - Extracts and interpolates marker data,
      - Detrends and applies a bandpass filter,
      - Runs PCA analysis,
      - Optionally plots the PCA projection.
    """
    cols_a = get_marker_columns(structural_features, marker_a)
    cols_b = get_marker_columns(structural_features, marker_b)
    if debug:
        logger.debug("Patient %s, hand %s: marker_a (%s) columns: %s", p.patient_id, hand, marker_a, cols_a)
        logger.debug("Patient %s, hand %s: marker_b (%s) columns: %s", p.patient_id, hand, marker_b, cols_b)
    
    if len(cols_a) == 0 or len(cols_b) == 0 or (len(cols_a) != len(cols_b)):
        logger.debug("Mismatch or missing coordinates for %s hand for patient %s.", hand, p.patient_id)
        return None
    
    if time_periods and hand in time_periods and len(time_periods[hand]) > 0:
        key_names = [cols_a[0], cols_b[0]]
        start_frame = min([time_periods[hand].get(key, [0, 0])[0] for key in key_names])
        end_frame   = max([time_periods[hand].get(key, [0, structural_features.shape[0]])[1] for key in key_names])
    else:
        start_frame = 0
        end_frame = structural_features.shape[0]
    
    if debug:
        logger.debug("Patient %s, hand %s: Frame window %d to %d", p.patient_id, hand, start_frame, end_frame)
    
    try:
        data_a = structural_features.loc[:, cols_a].interpolate().iloc[start_frame:end_frame].dropna()
        logger.debug("Patient %s, hand %s: data_a has %d frames.", p.patient_id, hand, data_a.shape[0])
    except Exception as e:
        logger.debug("Error accessing %s columns for patient %s, hand %s: %s", marker_a, p.patient_id, hand, e)
        return None
    try:
        data_b = structural_features.loc[:, cols_b].interpolate().iloc[start_frame:end_frame].dropna()
        logger.debug("Patient %s, hand %s: data_b has %d frames.", p.patient_id, hand, data_b.shape[0])
    except Exception as e:
        logger.debug("Error accessing %s columns for patient %s, hand %s: %s", marker_b, p.patient_id, hand, e)
        return None
    
    common_index = data_a.index.intersection(data_b.index)
    data_a = data_a.loc[common_index]
    data_b = data_b.loc[common_index]
    if data_a.empty or data_b.empty:
        logger.debug("No overlapping data for patient %s, hand %s.", p.patient_id, hand)
        return None

    centroid = (data_a.values + data_b.values) / 2.0
    centroid_detrended = np.apply_along_axis(signal.detrend, 0, centroid)
    
    filtered = np.zeros_like(centroid_detrended)
    for i in range(centroid_detrended.shape[1]):
        filtered[:, i] = butter_bandpass_filter(centroid_detrended[:, i], 3, 12, fs, order=5)
    
    features, projection, principal_component = pca_tremor_analysis(filtered, fs)
    
    if debug:
        logger.debug("Patient %s, hand %s: PCA produced features: %s", p.patient_id, hand, features)
    
    if plot:
        plt.figure()
        plt.plot(projection)
        plt.title(f"{p.patient_id}_{hand}_{feature_prefix}")
        if save_plots:
            plt.savefig(os.path.join(PLOTS_DIR, f'pca_{feature_prefix}_{p.patient_id}_{hand}.svg'))
        plt.close()
    
    return features

#############################################
# Feature Extraction Functions via PCA
#############################################
def extract_proximal_arm_tremor_features(pc, plot=False, save_plots=False, debug=DEBUG) -> pd.DataFrame:
    """
    Extract proximal arm tremor features (shoulder and elbow markers) using PCA.
    Expects markers: right_shoulder/right_elbow and left_shoulder/left_elbow.
    Uses patient.proximal_tremor_features.
    """
    features_df = pd.DataFrame(index=pc.get_patient_ids())
    logger.info("Extracting proximal arm tremor features using PCA ...")
    
    for p in tqdm(pc.patients, total=len(pc.patients)):
        fs = p.sampling_frequency
        if not hasattr(p, 'proximal_tremor_features') or p.proximal_tremor_features is None or p.proximal_tremor_features.empty:
            logger.debug("Patient %s has no proximal tremor features.", p.patient_id)
            continue
        
        proximal_features = normalize_multiindex_columns(p.proximal_tremor_features)
        if debug:
            logger.debug("Patient %s proximal features loaded. Columns: %s", 
                         p.patient_id, proximal_features.columns.tolist())
        time_periods = p.hand_time_periods if hasattr(p, 'hand_time_periods') else None
        hands = check_hand_(p)
        
        for hand in hands:
            if hand == 'right':
                marker_a = "right_shoulder"
                marker_b = "right_elbow"
            else:
                marker_a = "left_shoulder"
                marker_b = "left_elbow"
            
            if debug:
                logger.debug("Patient %s, hand %s: Extracting proximal features using markers %s and %s", 
                             p.patient_id, hand, marker_a, marker_b)
            
            features = _extract_tremor_features_for_marker_pair(p, fs, proximal_features,
                                                                 marker_a, marker_b, hand,
                                                                 time_periods, debug, plot, save_plots,
                                                                 feature_prefix="proximal_arm")
            if features:
                for key, value in features.items():
                    col_name = f"{key}_proximal_arm_{hand}"
                    features_df.loc[p.patient_id, col_name] = value
            else:
                logger.debug("No proximal arm features returned for patient %s, hand %s.", p.patient_id, hand)
    return features_df

def extract_distal_arm_tremor_features(pc, plot=False, save_plots=False, debug=DEBUG) -> pd.DataFrame:
    """
    Extract distal arm tremor features (elbow and wrist markers) using PCA.
    Expects markers: right_elbow/right_wrist and left_elbow/left_wrist.
    Uses patient.distal_tremor_features.
    """
    features_df = pd.DataFrame(index=pc.get_patient_ids())
    logger.info("Extracting distal arm tremor features using PCA ...")
    
    for p in tqdm(pc.patients, total=len(pc.patients)):
        fs = p.sampling_frequency
        if not hasattr(p, 'distal_tremor_features') or p.distal_tremor_features is None or p.distal_tremor_features.empty:
            logger.debug("Patient %s has no distal tremor features.", p.patient_id)
            continue
        
        distal_features = normalize_multiindex_columns(p.distal_tremor_features)
        if debug:
            logger.debug("Patient %s distal features loaded. Columns: %s", 
                         p.patient_id, distal_features.columns.tolist())
        time_periods = p.hand_time_periods if hasattr(p, 'hand_time_periods') else None
        hands = check_hand_(p)
        
        for hand in hands:
            if hand == 'right':
                marker_a = "right_elbow"
                marker_b = "right_wrist"
            else:
                marker_a = "left_elbow"
                marker_b = "left_wrist"
            
            if debug:
                logger.debug("Patient %s, hand %s: Extracting distal features using markers %s and %s", 
                             p.patient_id, hand, marker_a, marker_b)
            
            features = _extract_tremor_features_for_marker_pair(p, fs, distal_features,
                                                                 marker_a, marker_b, hand,
                                                                 time_periods, debug, plot, save_plots,
                                                                 feature_prefix="distal_arm")
            if features:
                for key, value in features.items():
                    col_name = f"{key}_distal_arm_{hand}"
                    features_df.loc[p.patient_id, col_name] = value
            else:
                logger.debug("No distal arm features returned for patient %s, hand %s.", p.patient_id, hand)
    return features_df

def extract_fingers_tremor_features(pc, plot=False, save_plots=False, debug=DEBUG) -> pd.DataFrame:
    """
    Extract fingers tremor features via PCA.
    Expects markers: index_finger_tip and middle_finger_tip for each hand.
    Uses patient.fingers_tremor_features.
    """
    features_df = pd.DataFrame(index=pc.get_patient_ids())
    logger.info("Extracting fingers tremor features using PCA ...")
    
    for p in tqdm(pc.patients, total=len(pc.patients)):
        fs = p.sampling_frequency
        if not hasattr(p, 'fingers_tremor_features') or p.fingers_tremor_features is None or p.fingers_tremor_features.empty:
            logger.debug("Patient %s has no fingers tremor features.", p.patient_id)
            continue
        
        fingers_features = normalize_multiindex_columns(p.fingers_tremor_features)
        if debug:
            logger.debug("Patient %s fingers features loaded. Columns: %s", 
                         p.patient_id, fingers_features.columns.tolist())
        time_periods = p.hand_time_periods if hasattr(p, 'hand_time_periods') else None
        hands = check_hand_(p)
        
        for hand in hands:
            if hand == 'right':
                marker_a = "index_finger_tip_right"
                marker_b = "middle_finger_tip_right"
            else:
                marker_a = "index_finger_tip_left"
                marker_b = "middle_finger_tip_left"
            
            if debug:
                logger.debug("Patient %s, hand %s: Extracting fingers features using markers %s and %s", 
                             p.patient_id, hand, marker_a, marker_b)
            
            features = _extract_tremor_features_for_marker_pair(p, fs, fingers_features,
                                                                 marker_a, marker_b, hand,
                                                                 time_periods, debug, plot, save_plots,
                                                                 feature_prefix="fingers")
            if features:
                for key, value in features.items():
                    col_name = f"{key}_fingers_{hand}"
                    features_df.loc[p.patient_id, col_name] = value
            else:
                logger.debug("No fingers tremor features returned for patient %s, hand %s.", p.patient_id, hand)
    return features_df
