# src/modules/tremor/normalization.py

import pandas as pd
import numpy as np

def compute_scaling_factor(df: pd.DataFrame, marker_a: str, marker_b: str) -> float:
    """Compute the average Euclidean distance between two markers."""
    try:
        xa, ya = df[(marker_a, 'x')], df[(marker_a, 'y')]
        xb, yb = df[(marker_b, 'x')], df[(marker_b, 'y')]
        distances = np.sqrt((xa - xb)**2 + (ya - yb)**2)
        return distances.mean()
    except KeyError:
        return 1.0

def compute_local_scaling_factors(df: pd.DataFrame) -> dict:
    """Computes local scaling factors for arm segments."""
    factors = {}
    factors["proximal_right"] = compute_scaling_factor(df, "right_shoulder", "right_elbow")
    factors["proximal_left"] = compute_scaling_factor(df, "left_shoulder", "left_elbow")
    factors["distal_right"] = compute_scaling_factor(df, "right_elbow", "right_wrist")
    factors["distal_left"] = compute_scaling_factor(df, "left_elbow", "left_wrist")
    # Add finger scaling if hand landmarks are present
    if ('index_finger_mcp_right', 'x') in df.columns:
        factors["finger_right"] = compute_scaling_factor(df, "index_finger_mcp_right", "index_finger_tip_right")
    if ('index_finger_mcp_left', 'x') in df.columns:
        factors["finger_left"] = compute_scaling_factor(df, "index_finger_mcp_left", "index_finger_tip_left")
    return factors

def zscore_normalization(df: pd.DataFrame) -> pd.DataFrame:
    """Applies z-score normalization column-wise."""
    normalized_df = df.copy()
    for col in normalized_df.columns:
        series = normalized_df[col]
        mean, std = series.mean(), series.std()
        if std != 0:
            normalized_df[col] = (series - mean) / std
        else:
            normalized_df[col] = series - mean
    return normalized_df

def normalize_pose_data(pc):
    """Normalizes the pose estimation data for each patient in the collection."""
    for patient in pc.patients:
        df = patient.pose_estimation
        if df is not None and not df.empty:
            local_factors = compute_local_scaling_factors(df)
            
            # This is a simplified local normalization. A full implementation would map each marker.
            # For this pipeline, we will just apply z-score, which is most critical.
            df = zscore_normalization(df)
            
            patient.pose_estimation = df
    return pc