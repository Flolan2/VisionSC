#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tremor_feature_extraction.py
Created on 22.03.25
Author: Florian Lange

This module contains functions for extracting tremor features from structural data using PCA.
It leverages functions from the signal_preprocessing module and utility functions from tremor_utils.
"""

import os
import logging
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from .tremor_utils import (
    identify_active_time_period,
    check_hand_,
    meanfilt,
    normalize_multiindex_columns,
    get_marker_columns,
)
from .tremor_signal_analysis import butter_bandpass_filter, pca_tremor_analysis

# Global output directory for plots.
PLOTS_DIR = "./plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

logger = logging.getLogger(__name__)

#############################################
# Active Time Period Assignment
#############################################
def assign_hand_time_periods(pc):
    """
    Assign active time periods for each patient.
    Chooses the first available structural data source, normalizes its columns,
    and computes the active time periods.
    """
    logger.info("Extracting active time periods of hands ...")
    for p in tqdm(pc.patients, total=len(pc.patients)):
        fs = p.sampling_frequency
        structural_data = None
        if hasattr(p, 'proximal_tremor_features') and not p.proximal_tremor_features.empty:
            structural_data = p.proximal_tremor_features
        elif hasattr(p, 'distal_tremor_features') and not p.distal_tremor_features.empty:
            structural_data = p.distal_tremor_features
        elif hasattr(p, 'fingers_tremor_features') and not p.fingers_tremor_features.empty:
            structural_data = p.fingers_tremor_features
        elif hasattr(p, 'structural_features') and not p.structural_features.empty:
            structural_data = p.structural_features
        
        if structural_data is not None:
            structural_data = normalize_multiindex_columns(structural_data)
            hands = check_hand_(p)
            p.hand_time_periods = identify_active_time_period(structural_data, fs) if len(hands) > 1 else None
        else:
            p.hand_time_periods = None
    return pc

#############################################
# Helper for Marker-Pair Feature Extraction
#############################################
def _extract_tremor_features_for_marker_pair(p, fs, structural_features: pd.DataFrame,
                                             marker_a: str, marker_b: str,
                                             hand: str, time_periods: dict,
                                             plot: bool, save_plots: bool,
                                             feature_prefix: str) -> dict:
    """
    Extract tremor features for a given patient and pair of markers.
    Performs interpolation, detrending, bandpass filtering, and PCA analysis.
    """
    cols_a = get_marker_columns(structural_features, marker_a)
    cols_b = get_marker_columns(structural_features, marker_b)
    if not cols_a or not cols_b or (len(cols_a) != len(cols_b)):
        return None
    
    if time_periods and hand in time_periods and time_periods[hand]:
        key_names = [cols_a[0], cols_b[0]]
        start_frame = min(
            time_periods[hand].get(key_names[0], [0, 0])[0],
            time_periods[hand].get(key_names[1], [0, 0])[0]
        )
        end_frame = max(
            time_periods[hand].get(key_names[0], [0, structural_features.shape[0]])[1],
            time_periods[hand].get(key_names[1], [0, structural_features.shape[0]])[1]
        )
    else:
        start_frame, end_frame = 0, structural_features.shape[0]
    
    data_a = structural_features.loc[:, cols_a].interpolate().iloc[start_frame:end_frame].dropna()
    data_b = structural_features.loc[:, cols_b].interpolate().iloc[start_frame:end_frame].dropna()
    
    common_index = data_a.index.intersection(data_b.index)
    data_a = data_a.loc[common_index]
    data_b = data_b.loc[common_index]
    if data_a.empty or data_b.empty:
        return None

    centroid = (data_a.values + data_b.values) / 2.0
    centroid_detrended = np.apply_along_axis(signal.detrend, 0, centroid)
    
    filtered = np.zeros_like(centroid_detrended)
    for i in range(centroid_detrended.shape[1]):
        filtered[:, i] = butter_bandpass_filter(centroid_detrended[:, i], 3, 12, fs, order=5)
    
    features, projection, principal_component = pca_tremor_analysis(filtered, fs)
    
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
def extract_proximal_arm_tremor_features(pc, plot=False, save_plots=False) -> pd.DataFrame:
    """
    Extract proximal arm tremor features (shoulder and elbow markers) using PCA.
    """
    features_df = pd.DataFrame(index=pc.get_patient_ids())
    logger.info("Extracting proximal arm tremor features using PCA ...")
    
    for p in tqdm(pc.patients, total=len(pc.patients)):
        fs = p.sampling_frequency
        if not hasattr(p, 'proximal_tremor_features') or p.proximal_tremor_features.empty:
            continue
        
        proximal_features = normalize_multiindex_columns(p.proximal_tremor_features)
        time_periods = getattr(p, 'hand_time_periods', None)
        hands = check_hand_(p)
        
        for hand in hands:
            if hand == 'right':
                marker_a, marker_b = "right_shoulder", "right_elbow"
            else:
                marker_a, marker_b = "left_shoulder", "left_elbow"
            
            features = _extract_tremor_features_for_marker_pair(
                p, fs, proximal_features, marker_a, marker_b, hand,
                time_periods, plot, save_plots, feature_prefix="proximal_arm"
            )
            if features:
                for key, value in features.items():
                    features_df.loc[p.patient_id, f"{key}_proximal_arm_{hand}"] = value
    return features_df

def extract_distal_arm_tremor_features(pc, plot=False, save_plots=False) -> pd.DataFrame:
    """
    Extract distal arm tremor features (elbow and wrist markers) using PCA.
    """
    features_df = pd.DataFrame(index=pc.get_patient_ids())
    logger.info("Extracting distal arm tremor features using PCA ...")
    
    for p in tqdm(pc.patients, total=len(pc.patients)):
        fs = p.sampling_frequency
        if not hasattr(p, 'distal_tremor_features') or p.distal_tremor_features.empty:
            continue
        
        distal_features = normalize_multiindex_columns(p.distal_tremor_features)
        time_periods = getattr(p, 'hand_time_periods', None)
        hands = check_hand_(p)
        
        for hand in hands:
            if hand == 'right':
                marker_a, marker_b = "right_elbow", "right_wrist"
            else:
                marker_a, marker_b = "left_elbow", "left_wrist"
            
            features = _extract_tremor_features_for_marker_pair(
                p, fs, distal_features, marker_a, marker_b, hand,
                time_periods, plot, save_plots, feature_prefix="distal_arm"
            )
            if features:
                for key, value in features.items():
                    features_df.loc[p.patient_id, f"{key}_distal_arm_{hand}"] = value
    return features_df

def extract_fingers_tremor_features(pc, plot=False, save_plots=False) -> pd.DataFrame:
    """
    Extract fingers tremor features via PCA.
    """
    features_df = pd.DataFrame(index=pc.get_patient_ids())
    logger.info("Extracting fingers tremor features using PCA ...")
    
    for p in tqdm(pc.patients, total=len(pc.patients)):
        fs = p.sampling_frequency
        if not hasattr(p, 'fingers_tremor_features') or p.fingers_tremor_features.empty:
            continue
        
        fingers_features = normalize_multiindex_columns(p.fingers_tremor_features)
        time_periods = getattr(p, 'hand_time_periods', None)
        hands = check_hand_(p)
        
        for hand in hands:
            if hand == 'right':
                marker_a, marker_b = "index_finger_tip_right", "middle_finger_tip_right"
            else:
                marker_a, marker_b = "index_finger_tip_left", "middle_finger_tip_left"
            
            features = _extract_tremor_features_for_marker_pair(
                p, fs, fingers_features, marker_a, marker_b, hand,
                time_periods, plot, save_plots, feature_prefix="fingers"
            )
            if features:
                for key, value in features.items():
                    features_df.loc[p.patient_id, f"{key}_fingers_{hand}"] = value
    return features_df
