#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 22 19:08:02 2025

@author: Lange_L
"""

"""
normalization.py

This module provides functionality to perform both global and local normalization on pose estimation data.
It computes scaling factors based on anatomical landmarks so that tremor amplitudes become relative (dimensionless)
measures. The module includes:

  - compute_scaling_factor: Computes the average Euclidean distance between two markers.
  - compute_local_scaling_factors: Computes scaling factors for different body parts (fingers, proximal arm, distal arm).
    For fingers, it uses the distance from the MCP joint to the index fingertip.
  - apply_local_normalization: Scales the marker coordinates by the appropriate local scaling factor.
  - zscore_normalization: (Optional) Applies z-score normalization columnwise.
  - normalize_pose_data: Applies local (and optionally global and z-score) normalization to each patient’s pose data.
"""

import pandas as pd
import numpy as np

def compute_scaling_factor(df: pd.DataFrame, marker_a: str, marker_b: str) -> float:
    """
    Compute the average Euclidean distance between two markers (using x and y coordinates)
    across all frames in the pose estimation DataFrame.

    Parameters:
        df (pd.DataFrame): Pose estimation DataFrame with MultiIndex columns (marker, coordinate).
        marker_a (str): Name of the first marker.
        marker_b (str): Name of the second marker.

    Returns:
        float: The average Euclidean distance between the two markers.
               Returns 1.0 if the required columns are not found to avoid division by zero.
    """
    try:
        # Extract x and y for each marker
        xa = df[(marker_a, 'x')]
        ya = df[(marker_a, 'y')]
        xb = df[(marker_b, 'x')]
        yb = df[(marker_b, 'y')]
        # Compute Euclidean distance for each frame
        distances = np.sqrt((xa - xb) ** 2 + (ya - yb) ** 2)
        # Return the mean distance over all frames
        return distances.mean()
    except KeyError:
        # If markers not found, return a default scaling factor of 1.0
        return 1.0

def compute_local_scaling_factors(df: pd.DataFrame) -> dict:
    """
    Computes local scaling factors based on anatomical landmarks present in the DataFrame.
    For each side, it calculates:
      - Finger scaling: distance between the MCP joint and the index fingertip.
      - Proximal arm scaling: distance between shoulder and elbow.
      - Distal arm scaling: distance between elbow and wrist.

    Parameters:
        df (pd.DataFrame): Pose estimation DataFrame with MultiIndex columns.

    Returns:
        dict: A dictionary with scaling factors keyed by body part and side.
              Example keys: "finger_right", "finger_left", "proximal_right", "proximal_left",
                            "distal_right", "distal_left".
    """
    factors = {}
    # Use the MCP-to-index fingertip distance for finger scaling.
    factors["finger_right"] = compute_scaling_factor(df, "index_finger_mcp_right", "index_finger_tip_right")
    factors["finger_left"]  = compute_scaling_factor(df, "index_finger_mcp_left", "index_finger_tip_left")
    
    # Proximal arm lengths (shoulder to elbow)
    factors["proximal_right"] = compute_scaling_factor(df, "right_shoulder", "right_elbow")
    factors["proximal_left"]  = compute_scaling_factor(df, "left_shoulder", "left_elbow")
    
    # Distal arm lengths (elbow to wrist)
    factors["distal_right"] = compute_scaling_factor(df, "right_elbow", "right_wrist")
    factors["distal_left"]  = compute_scaling_factor(df, "left_elbow", "left_wrist")
    return factors

def apply_local_normalization(df: pd.DataFrame, scaling_factors: dict, global_scaling: float = 1.0) -> pd.DataFrame:
    """
    Applies local normalization to the pose estimation DataFrame.
    Each marker coordinate is divided by the corresponding local scaling factor,
    and optionally multiplied by a global scaling factor if provided.
    
    The mapping is as follows:
      - Finger markers ("index_finger_tip_*", "index_finger_mcp_*", "middle_finger_tip_*") are scaled by the finger factor.
      - Shoulder markers are scaled by the proximal factor.
      - Elbow markers are scaled by the proximal factor.
      - Wrist markers are scaled by the distal factor.
      
    Parameters:
        df (pd.DataFrame): Pose estimation DataFrame with MultiIndex columns.
        scaling_factors (dict): Dictionary of local scaling factors.
        global_scaling (float): Additional global scaling factor (default 1.0).
    
    Returns:
        pd.DataFrame: New DataFrame with locally normalized coordinates.
    """
    normalized_df = df.copy()
    
    # Mapping from marker name to corresponding scaling factor key.
    marker_to_factor = {
        "index_finger_tip_right": "finger_right",
        "index_finger_mcp_right": "finger_right",
        "middle_finger_tip_right": "finger_right",
        "index_finger_tip_left": "finger_left",
        "index_finger_mcp_left": "finger_left",
        "middle_finger_tip_left": "finger_left",
        "right_shoulder": "proximal_right",
        "left_shoulder": "proximal_left",
        "right_elbow": "proximal_right",   # Using proximal factor for elbow
        "left_elbow": "proximal_left",
        "right_wrist": "distal_right",
        "left_wrist": "distal_left"
    }
    
    # Iterate over each column in the MultiIndex DataFrame
    for marker, coord in normalized_df.columns:
        factor_key = marker_to_factor.get(marker, None)
        if factor_key is not None:
            factor = scaling_factors.get(factor_key, 1.0)
            # Avoid division by zero; if factor is zero, skip normalization for that marker.
            if factor != 0:
                normalized_df[(marker, coord)] = (normalized_df[(marker, coord)] / factor) * global_scaling
    return normalized_df

def zscore_normalization(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies z-score normalization to each column of the DataFrame.
    This centers the data (subtracts the mean) and scales it to unit variance (divides by the standard deviation).

    Parameters:
        df (pd.DataFrame): The pose estimation DataFrame with a MultiIndex.

    Returns:
        pd.DataFrame: New DataFrame with z-score normalized columns.
    """
    normalized_df = df.copy()
    for col in normalized_df.columns:
        series = normalized_df[col]
        mean = series.mean()
        std = series.std()
        if std != 0:
            normalized_df[col] = (series - mean) / std
        else:
            normalized_df[col] = series - mean
    return normalized_df

def normalize_pose_data(pc, global_scaling_factor: float = 1.0, use_local: bool = True, use_zscore: bool = True):
    """
    Normalizes the pose estimation data for each patient in the PatientCollection.
    
    If use_local is True, it computes local scaling factors for each patient based on anatomical landmarks
    and applies them to the data. Then, if use_zscore is True, it applies z-score normalization.
    If use_local is False, it can simply apply a global scaling (if desired) and z-score normalization.

    This function updates each patient's 'pose_estimation' attribute.

    Parameters:
        pc: PatientCollection, expected to have an attribute 'patients' (an iterable of patient objects)
            where each patient has a 'pose_estimation' DataFrame.
        global_scaling_factor (float): Global scaling factor to apply after local normalization (default 1.0).
        use_local (bool): Whether to apply local (anatomical) normalization.
        use_zscore (bool): Whether to apply z-score normalization after scaling.

    Returns:
        Updated PatientCollection with normalized pose estimation data.
    """
    for patient in pc.patients:
        if hasattr(patient, "pose_estimation"):
            df = patient.pose_estimation
            # Apply local normalization if requested
            if use_local:
                local_factors = compute_local_scaling_factors(df)
                df = apply_local_normalization(df, local_factors, global_scaling=global_scaling_factor)
            else:
                # If not using local normalization, simply apply global scaling.
                df = df * global_scaling_factor
            
            # Optionally apply z-score normalization
            if use_zscore:
                df = zscore_normalization(df)
            
            patient.pose_estimation = df
    return pc
