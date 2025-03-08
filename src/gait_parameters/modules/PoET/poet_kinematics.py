import pandas as pd
from tqdm import tqdm

# Import check_hand_ from your poet_utils module.
from .poet_utils import check_hand_

def extract_marker_data(pose_estimation, marker_name):
    """
    Given the multi-index DataFrame `pose_estimation` and a marker name,
    this function extracts the available coordinate data (x, y, and z if present)
    and returns a dictionary mapping new column names to the corresponding series.
    """
    data = {}
    for coord in ['x', 'y', 'z']:
        if (marker_name, coord) in pose_estimation.columns:
            new_col = f"marker_{marker_name}_{coord}"
            data[new_col] = pose_estimation[(marker_name, coord)]
    return data

def extract_tremor(pc):
    # Define markers for general tremor analysis.
    marker_features = {
        'right': [
            'index_finger_tip_right',
            'middle_finger_tip_right',
            'right_elbow'
        ],
        'left': [
            'index_finger_tip_left',
            'middle_finger_tip_left',
            'left_elbow'
        ]
    }

    print('Extracting tremor ... ')
    for patient in pc.patients:
        hands = check_hand_(patient)
        pose_estimation = patient.pose_estimation
        # Create a new DataFrame for the computed kinematic features,
        # but do not overwrite the original data.
        kinematic_features = pd.DataFrame(index=pose_estimation.index)

        for hand in hands:
            for marker in marker_features[hand]:
                marker_data = extract_marker_data(pose_estimation, marker)
                for col_name, series in marker_data.items():
                    kinematic_features[col_name] = series

        # Store kinematic features in a separate attribute.
        patient.kinematic_features = kinematic_features

    return pc


def extract_kinematic_tremor(pc):
    marker_features = {
        'right': ['index_finger_tip_right'],
        'left': ['index_finger_tip_left']
    }

    print('Extracting intention tremor ... ')
    for patient in tqdm(pc.patients, total=len(pc.patients)):
        hands = check_hand_(patient)
        pose_estimation = patient.pose_estimation
        structural_features = pd.DataFrame(index=pose_estimation.index)

        for hand in hands:
            for marker in marker_features[hand]:
                marker_data = extract_marker_data(pose_estimation, marker)
                for col_name, series in marker_data.items():
                    structural_features[col_name] = series

        patient.structural_features = structural_features

    return pc

def extract_postural_tremor(pc):
    marker_features = {
        'right': ['middle_finger_tip_right'],
        'left': ['middle_finger_tip_left']
    }

    print('Extracting postural tremor ... ')
    for patient in tqdm(pc.patients, total=len(pc.patients)):
        hands = check_hand_(patient)
        pose_estimation = patient.pose_estimation
        structural_features = pd.DataFrame(index=pose_estimation.index)

        for hand in hands:
            for marker in marker_features[hand]:
                marker_data = extract_marker_data(pose_estimation, marker)
                for col_name, series in marker_data.items():
                    structural_features[col_name] = series

        patient.structural_features = structural_features

    return pc

def extract_proximal_tremor(pc):
    """
    Extract proximal tremor features using shoulder and elbow markers.
    For the right hand: uses 'right_shoulder' and 'right_elbow'.
    For the left hand: uses 'left_shoulder' and 'left_elbow'.
    """
    marker_features = {
        'right': ['right_shoulder', 'right_elbow'],
        'left': ['left_shoulder', 'left_elbow']
    }

    print('Extracting proximal tremor ... ')
    for patient in tqdm(pc.patients, total=len(pc.patients)):
        hands = check_hand_(patient)
        pose_estimation = patient.pose_estimation
        structural_features = pd.DataFrame(index=pose_estimation.index)
        
        for hand in hands:
            for marker in marker_features[hand]:
                marker_data = extract_marker_data(pose_estimation, marker)
                for col_name, series in marker_data.items():
                    structural_features[col_name] = series
        patient.structural_features = structural_features

    return pc

def extract_distal_tremor(pc):
    """
    Extract distal tremor features using elbow and wrist markers.
    For the right hand: uses 'right_elbow' and 'right_wrist'.
    For the left hand: uses 'left_elbow' and 'left_wrist'.
    """
    marker_features = {
        'right': ['right_elbow', 'right_wrist'],
        'left': ['left_elbow', 'left_wrist']
    }

    print('Extracting distal tremor ... ')
    for patient in tqdm(pc.patients, total=len(pc.patients)):
        hands = check_hand_(patient)
        pose_estimation = patient.pose_estimation
        structural_features = pd.DataFrame(index=pose_estimation.index)
        
        for hand in hands:
            for marker in marker_features[hand]:
                marker_data = extract_marker_data(pose_estimation, marker)
                for col_name, series in marker_data.items():
                    structural_features[col_name] = series
        patient.structural_features = structural_features

    return pc
