import numpy as np
import pandas as pd
import logging
from sklearn.decomposition import PCA
import os
import json

from modules.gait.my_utils.helpers import detect_extremas
# Update the plotting import to the new combined function
from modules.gait.my_utils.plotting import plot_combined_extremas_and_toe, plot_extrema_frames

def get_project_root():
    """
    Returns the absolute path two levels above this file.
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)

def get_plots_dir():
    """
    Always returns the plots directory inside the external Output folder.
    The external Output folder is defined as one level above the project root.
    """
    project_root = get_project_root()
    external_output = os.path.join(os.path.abspath(os.path.join(project_root, "..")), "Output")
    plots_dir = os.path.join(external_output, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir

# Configure logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.hasHandlers():
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

class EventDetector:
    """
    Detects heel strike (HS) and toe-off (TO) events from pose data.
    """
    
    def __init__(self, input_path, algorithm="zeni", make_plot=True, frame_rate=25, window_size=100, step_size=50, config=None, **kwargs):
        self.input_path = input_path 
        # Hardcode the plots directory to always be the external Output/plots folder.
        self.plots_dir = get_plots_dir()
        os.makedirs(self.plots_dir, exist_ok=True)
        self.algorithm = algorithm
        self.make_plot = make_plot
        self.frame_rate = frame_rate
        self.window_size = window_size
        self.step_size = step_size
        self.config = config or {}
        
        # Log the frame rate being used.
        logger.info(f"EventDetector initialized with frame_rate: {self.frame_rate} fps")
    
    def detect_heel_toe_events(self, pose_data):
        # Log the column names of the incoming pose_data DataFrame for debugging.
        logger.debug("Pose data columns in detect_heel_toe_events: {}".format(pose_data.columns))
        
        try:
            rotation_angles = compute_framewise_rotation_angles(
                pose_data,
                marker="sacrum",
                window_size=self.window_size,
                step_size=self.step_size
            )
        except Exception as e:
            logger.exception("Error computing framewise rotation angles: %s", str(e))
            raise

        try:
            rotated_pose_data = rotate_pose_data_framewise(pose_data, rotation_angles)
        except Exception as e:
            logger.exception("Error rotating pose data: %s", str(e))
            raise

        try:
            if self.algorithm == "zeni":
                events = self._detect_events_zeni(rotated_pose_data)
            elif self.algorithm == "dewitt":
                events = self._detect_events_dewitt(rotated_pose_data)
            elif self.algorithm == "hreljac":
                events = self._detect_events_hreljac(rotated_pose_data)
            else:
                raise ValueError("Unsupported algorithm")
        except Exception as e:
            logger.exception("Error in event detection using algorithm %s: %s", self.algorithm, str(e))
            raise
        return events

    def _detect_events_zeni(self, pose_data):
        all_forward_movement = {}
        all_extrema_data = {}
        event_extrema_data = {}
        
        foot_landmarks = {
            "HS_left": "left_heel",
            "HS_right": "right_heel",
            "TO_left": "left_foot_index",
            "TO_right": "right_foot_index"
        }
    
        for landmark_name, landmark in foot_landmarks.items():
            try:
                forward_movement = pose_data[(landmark, 'z')] - pose_data[('sacrum', 'z')]
                all_forward_movement[landmark_name] = forward_movement
            except Exception as e:
                logger.exception("Error computing forward movement for landmark %s: %s", landmark, str(e))
                continue
    
            try:
                peaks, valleys = detect_extremas(forward_movement)
            except Exception as e:
                logger.exception("Error detecting extremas for landmark %s: %s", landmark, str(e))
                peaks, valleys = np.array([]), np.array([])
    
            try:
                all_extrema_data[landmark_name] = {
                    "peaks": peaks / self.frame_rate if peaks.size else peaks,
                    "valleys": valleys / self.frame_rate if valleys.size else valleys
                }
            except Exception as e:
                logger.exception("Error converting extremas for landmark %s: %s", landmark, str(e))
                all_extrema_data[landmark_name] = {"peaks": peaks, "valleys": valleys}
                
            try:
                if "HS" in landmark_name:
                    idx = peaks  # Heel Strike
                else:
                    idx = valleys  # Toe-Off
                event_extrema_data[landmark_name] = idx / self.frame_rate
            except Exception as e:
                logger.exception("Error converting indices for landmark %s: %s", landmark, str(e))
                event_extrema_data[landmark_name] = np.array([])
    
        try:
            max_length = max(len(v) for v in event_extrema_data.values() if isinstance(v, (list, np.ndarray)))
        except Exception as e:
            logger.exception("Error computing maximum length of events: %s", str(e))
            max_length = 0
    
        try:
            events = pd.DataFrame({
                key: pd.Series(list(values) + [np.nan] * (max_length - len(values)))
                for key, values in event_extrema_data.items()
            })
            # Log the detected event times for heel strikes.
            logger.debug("Detected HS_left events: {}".format(events.get("HS_left")))
            logger.debug("Detected HS_right events: {}".format(events.get("HS_right")))
        except Exception as e:
            logger.exception("Error creating events DataFrame: %s", str(e))
            events = pd.DataFrame()
        return events

    def _detect_events_dewitt(self, pose_data):
        logger.error("Dewitt event detection not implemented yet.")
        raise NotImplementedError("Dewitt event detection not implemented yet.")

    def _detect_events_hreljac(self, pose_data):
        logger.error("Hreljac event detection not implemented yet.")
        raise NotImplementedError("Hreljac event detection not implemented yet.")


def compute_framewise_rotation_angles(pose_data, marker="sacrum", window_size=100, step_size=50):
    sliding_angles = determine_gait_direction_sliding_window(pose_data, marker, window_size, step_size)
    if not sliding_angles:
        logger.warning("No sliding angles computed; returning zero angles for all frames.")
        return np.zeros(len(pose_data))
    centers, window_angles = zip(*sliding_angles)
    centers = np.array(centers)
    window_angles = np.array(window_angles)
    num_frames = len(pose_data)
    frame_indices = np.arange(num_frames)
    framewise_angles = np.interp(frame_indices, centers, window_angles)
    return framewise_angles


def determine_gait_direction_sliding_window(pose_data, marker="sacrum", window_size=100, step_size=50):
    angles = []
    num_frames = len(pose_data)
    for start in range(0, num_frames - window_size + 1, step_size):
        end = start + window_size
        window_data = pose_data.iloc[start:end]
        try:
            x_coords = window_data[(marker, 'x')].to_numpy()
            z_coords = window_data[(marker, 'z')].to_numpy()
        except Exception as e:
            logger.exception("Error extracting coordinates for marker %s in window %d-%d: %s", marker, start, end, str(e))
            continue
        positions = np.column_stack((x_coords, z_coords))
        
        try:
            pca = PCA(n_components=1)
            pca.fit(positions)
            principal_vector = pca.components_[0]
            angle = np.arctan2(principal_vector[0], principal_vector[1])
        except Exception as e:
            logger.exception("Error in PCA fitting for sliding window starting at index %d: %s", start, str(e))
            angle = 0
        
        window_center = start + window_size // 2
        angles.append((window_center, -angle))
    return angles


def rotate_pose_data_framewise(pose_data, rotation_angles):
    new_data = pose_data.copy()
    markers = pose_data.columns.get_level_values(0).unique()
    for marker in markers:
        if (marker, 'x') in pose_data.columns and (marker, 'z') in pose_data.columns:
            try:
                x_orig = pose_data[(marker, 'x')].to_numpy()
                z_orig = pose_data[(marker, 'z')].to_numpy()
                new_x = x_orig * np.cos(rotation_angles) - z_orig * np.sin(rotation_angles)
                new_z = x_orig * np.sin(rotation_angles) + z_orig * np.cos(rotation_angles)
                new_data[(marker, 'x')] = new_x
                new_data[(marker, 'z')] = new_z
            except Exception as e:
                logger.exception("Error rotating marker %s: %s", marker, str(e))
    return new_data
