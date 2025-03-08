import pandas as pd
from tqdm import tqdm

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

def extract_intention_tremor(pc):
    """
    Extract intention tremor features using index finger tip markers.
    For the right hand: uses 'index_finger_tip_right'.
    For the left hand: uses 'index_finger_tip_left'.
    """
    marker_features = {
        'right': ['index_finger_tip_right'],
        'left': ['index_finger_tip_left']
    }
    print('Extracting intention tremor ... ')
    for patient in tqdm(pc.patients, total=len(pc.patients)):
        hands = check_hand_(patient)
        pose_estimation = patient.pose_estimation
        intention_features = pd.DataFrame(index=pose_estimation.index)
        for hand in hands:
            for marker in marker_features[hand]:
                marker_data = extract_marker_data(pose_estimation, marker)
                for col_name, series in marker_data.items():
                    intention_features[col_name] = series
        # Store output in a dedicated attribute.
        patient.intention_tremor_features = intention_features
    return pc

def extract_postural_tremor(pc):
    """
    Extract postural tremor features using middle finger tip markers.
    For the right hand: uses 'middle_finger_tip_right'.
    For the left hand: uses 'middle_finger_tip_left'.
    """
    marker_features = {
        'right': ['middle_finger_tip_right'],
        'left': ['middle_finger_tip_left']
    }
    print('Extracting postural tremor ... ')
    for patient in tqdm(pc.patients, total=len(pc.patients)):
        hands = check_hand_(patient)
        pose_estimation = patient.pose_estimation
        postural_features = pd.DataFrame(index=pose_estimation.index)
        for hand in hands:
            for marker in marker_features[hand]:
                marker_data = extract_marker_data(pose_estimation, marker)
                for col_name, series in marker_data.items():
                    postural_features[col_name] = series
        # Store output in a dedicated attribute.
        patient.postural_tremor_features = postural_features
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
        proximal_features = pd.DataFrame(index=pose_estimation.index)
        for hand in hands:
            for marker in marker_features[hand]:
                marker_data = extract_marker_data(pose_estimation, marker)
                for col_name, series in marker_data.items():
                    proximal_features[col_name] = series
        # Save output separately.
        patient.proximal_tremor_features = proximal_features
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
        distal_features = pd.DataFrame(index=pose_estimation.index)
        for hand in hands:
            for marker in marker_features[hand]:
                marker_data = extract_marker_data(pose_estimation, marker)
                for col_name, series in marker_data.items():
                    distal_features[col_name] = series
        # Save output separately.
        patient.distal_tremor_features = distal_features
    return pc

def extract_fingers_tremor(pc):
    """
    Extract finger tremor features using finger tip markers.
    For the right hand: uses 'index_finger_tip_right' and 'middle_finger_tip_right'.
    For the left hand: uses 'index_finger_tip_left' and 'middle_finger_tip_left'.
    
    This function replicates the old functionality by explicitly extracting 
    the 'x' and 'y' coordinates for each marker and constructing new column names.
    The resulting DataFrame is stored in patient.fingers_tremor_features.
    """
    # Define markers for finger tremor extraction (only using finger tip markers and only x and y coordinates)
    marker_features = {
        'right': [
            ['index_finger_tip_right', 'x'], ['index_finger_tip_right', 'y'],
            ['middle_finger_tip_right', 'x'], ['middle_finger_tip_right', 'y']
        ],
        'left': [
            ['index_finger_tip_left', 'x'], ['index_finger_tip_left', 'y'],
            ['middle_finger_tip_left', 'x'], ['middle_finger_tip_left', 'y']
        ]
    }
    
    print('Extracting finger tremor ... ')
    from tqdm import tqdm  # ensure tqdm is imported
    for patient in tqdm(pc.patients, total=len(pc.patients)):
        # Determine which hand(s) are available for this patient.
        hands = check_hand_(patient)
        pose_estimation = patient.pose_estimation
        # Build a new DataFrame solely for finger tremor features.
        fingers_features = pd.DataFrame(index=pose_estimation.index)
        
        for hand in hands:
            features = marker_features.get(hand, [])
            for f in features:
                # Construct a column name in the form 'marker_<marker>_<coord>'
                col_name = 'marker_' + '_'.join(f)
                try:
                    # Directly assign the corresponding column from the multi-index DataFrame.
                    fingers_features[col_name] = pose_estimation[(f[0], f[1])]
                except KeyError:
                    # Log or print a warning if the expected marker is missing.
                    print(f"Warning: Marker {(f[0], f[1])} not found in pose estimation for patient {patient}")
                    
        # Store the computed finger tremor features in a dedicated attribute.
        patient.fingers_tremor_features = fingers_features
        
    return pc
