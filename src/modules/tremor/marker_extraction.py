# src/modules/tremor/marker_extraction.py

import pandas as pd
from .utils import check_hand_, get_marker_columns

def _extract_segment_data(patient, marker_map):
    """
    Helper function to extract data for a given set of markers, preserving the MultiIndex.
    It now gracefully handles markers that might not exist in the dataframe.
    """
    hands = check_hand_(patient)
    all_cols = []
    
    # Get a flat list of available markers from the dataframe columns
    available_markers = patient.pose_estimation.columns.get_level_values(0).unique()

    for hand in hands:
        # For each hand, get the list of requested markers
        for marker_name in marker_map.get(hand, []):
            # *** MODIFICATION START ***
            # Only proceed if the requested marker actually exists in the dataframe
            if marker_name in available_markers:
                all_cols.extend(get_marker_columns(patient.pose_estimation, marker_name))
            else:
                # This marker was not found in the pose data, skip it.
                pass
            # *** MODIFICATION END ***
    
    # If no relevant columns were found at all, return an empty DataFrame
    # with the same index to prevent downstream errors.
    if not all_cols:
        return pd.DataFrame(index=patient.pose_estimation.index)

    # Return a view of the original DataFrame with only the relevant columns
    return patient.pose_estimation[all_cols]

def extract_proximal_tremor(pc):
    """Extracts proximal tremor data (shoulder, elbow)."""
    marker_map = {'right': ['right_shoulder', 'right_elbow'], 'left': ['left_shoulder', 'left_elbow']}
    for patient in pc.patients:
        patient.proximal_tremor_features = _extract_segment_data(patient, marker_map)
    return pc

def extract_distal_tremor(pc):
    """Extracts distal tremor data (elbow, wrist)."""
    marker_map = {'right': ['right_elbow', 'right_wrist'], 'left': ['left_elbow', 'left_wrist']}
    for patient in pc.patients:
        patient.distal_tremor_features = _extract_segment_data(patient, marker_map)
    return pc

def extract_fingers_tremor(pc):
    """Extracts finger tremor data (index and middle finger tips)."""
    marker_map = {'right': ['index_finger_tip_right', 'middle_finger_tip_right'],
                  'left': ['index_finger_tip_left', 'middle_finger_tip_left']}
    for patient in pc.patients:
        patient.fingers_tremor_features = _extract_segment_data(patient, marker_map)
    return pc