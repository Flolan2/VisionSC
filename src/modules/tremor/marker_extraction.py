# src/modules/tremor/marker_extraction.py

import pandas as pd
from .utils import check_hand_

def extract_marker_data(pose_estimation, marker_name):
    """Extracts coordinate data for a given marker."""
    data = {}
    for coord in ['x', 'y', 'z']:
        if (marker_name, coord) in pose_estimation.columns:
            # Create a simple column name for the new DataFrame
            new_col = f"{marker_name}_{coord}"
            data[new_col] = pose_estimation[(marker_name, coord)]
    return data

def extract_proximal_tremor(pc):
    """Extracts proximal tremor features (shoulder, elbow)."""
    marker_map = {'right': ['right_shoulder', 'right_elbow'], 'left': ['left_shoulder', 'left_elbow']}
    for patient in pc.patients:
        hands = check_hand_(patient)
        features = pd.DataFrame(index=patient.pose_estimation.index)
        for hand in hands:
            for marker in marker_map[hand]:
                marker_data = extract_marker_data(patient.pose_estimation, marker)
                for col_name, series in marker_data.items():
                    features[col_name] = series
        patient.proximal_tremor_features = features
    return pc

def extract_distal_tremor(pc):
    """Extracts distal tremor features (elbow, wrist)."""
    marker_map = {'right': ['right_elbow', 'right_wrist'], 'left': ['left_elbow', 'left_wrist']}
    for patient in pc.patients:
        hands = check_hand_(patient)
        features = pd.DataFrame(index=patient.pose_estimation.index)
        for hand in hands:
            for marker in marker_map[hand]:
                marker_data = extract_marker_data(patient.pose_estimation, marker)
                for col_name, series in marker_data.items():
                    features[col_name] = series
        patient.distal_tremor_features = features
    return pc

def extract_fingers_tremor(pc):
    """Extracts finger tremor features (index and middle finger tips)."""
    marker_map = {'right': ['index_finger_tip_right', 'middle_finger_tip_right'],
                  'left': ['index_finger_tip_left', 'middle_finger_tip_left']}
    for patient in pc.patients:
        hands = check_hand_(patient)
        features = pd.DataFrame(index=patient.pose_estimation.index)
        for hand in hands:
            for marker in marker_map[hand]:
                # In this case, we need the multi-index for PCA, so we copy the columns directly
                for coord in ['x', 'y', 'z']:
                    if (marker, coord) in patient.pose_estimation.columns:
                        features[(marker, coord)] = patient.pose_estimation[(marker, coord)]
        patient.fingers_tremor_features = features
    return pc