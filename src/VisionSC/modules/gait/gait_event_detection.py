import numpy as np
import pandas as pd
import logging
from sklearn.decomposition import PCA
import os
import json

from modules.gait.my_utils.helpers import detect_extremas
# Import only the needed plot function (plot generation is now handled in gait_pipeline.py)
from modules.gait.my_utils.plotting import plot_combined_extremas_and_toe, butter_lowpass_filter

# Configure logger for this module
logger = logging.getLogger(__name__)

class EventDetector:
    """
    Detects heel strike (HS) and toe-off (TO) events from pose data using rotated coordinates.
    Modified to return both events DataFrame and the rotated pose data.
    """

    def __init__(self, input_path, algorithm="zeni", frame_rate=30, window_size=100, step_size=50, config=None, **kwargs):
        self.input_path = input_path
        self.algorithm = algorithm
        # self.make_plot = make_plot # Not used anymore
        self.frame_rate = frame_rate
        # Scale window/step size relative to frame rate? Or keep fixed duration?
        # Let's assume config values are in frames for now. If they were seconds, conversion needed.
        self.window_size = max(10, int(window_size)) # Ensure minimum reasonable window size
        self.step_size = max(1, int(step_size))     # Ensure step size is at least 1
        self.config = config or {}

        logger.info(f"EventDetector initialized with algorithm='{self.algorithm}', frame_rate={self.frame_rate} fps, window_size={self.window_size} frames, step_size={self.step_size} frames.")
        if self.frame_rate <= 0:
             logger.error(f"Invalid frame_rate ({self.frame_rate}) passed to EventDetector.")
             raise ValueError("Frame rate must be positive.")

    def detect_heel_toe_events(self, pose_data):
        """
        Detects gait events and returns both the events DataFrame and the rotated pose data used.
        """
        logger.debug(f"Pose data columns in detect_heel_toe_events: {pose_data.columns.tolist()}")

        required_markers = ['left_hip', 'right_hip', 'left_foot_index', 'right_foot_index', 'left_heel', 'right_heel', 'sacrum']
        present_markers = set(col[0] for col in pose_data.columns if isinstance(col, tuple)) # Check first level of MultiIndex
        if not all(marker in present_markers for marker in required_markers if marker != 'sacrum'):
             missing = [marker for marker in required_markers if marker != 'sacrum' and marker not in present_markers]
             logger.error(f"Missing required markers in pose_data for event detection: {missing}")
             raise ValueError(f"Pose data missing required markers: {missing}")

        try:
            logger.info("Computing framewise rotation angles...")
            rotation_angles = compute_framewise_rotation_angles(
                pose_data,
                marker="sacrum",
                window_size=self.window_size,
                step_size=self.step_size,
                frame_rate=self.frame_rate
            )
            logger.info("Finished computing rotation angles.")
        except Exception as e:
            logger.error(f"Error computing framewise rotation angles: {e}", exc_info=True)
            # Return empty results if rotation fails? Or raise? Let's raise.
            raise

        try:
            logger.info("Rotating pose data framewise...")
            rotated_pose_data = rotate_pose_data_framewise(pose_data, rotation_angles)
            logger.info("Finished rotating pose data.")
        except Exception as e:
            logger.error(f"Error rotating pose data: {e}", exc_info=True)
            # Return empty results if rotation fails?
            raise

        events = pd.DataFrame() # Initialize empty DataFrame
        try:
            logger.info(f"Detecting events using algorithm: {self.algorithm}")
            if self.algorithm == "zeni":
                events = self._detect_events_zeni(rotated_pose_data)
            else:
                logger.error(f"Unsupported event detection algorithm: {self.algorithm}")
                # Return empty events and the rotated data (might still be useful)
                return events, rotated_pose_data
                # raise ValueError(f"Unsupported algorithm: {self.algorithm}")
            logger.info("Finished event detection.")
        except Exception as e:
            logger.error(f"Error in event detection using algorithm {self.algorithm}: {e}", exc_info=True)
            # Return empty events but potentially valid rotated data
            return events, rotated_pose_data

        # *** RETURN BOTH events AND rotated_pose_data ***
        return events, rotated_pose_data

    def _detect_events_zeni(self, rotated_pose_data): # Changed argument name
        """
        Detects events using the Zeni method based on forward displacement extremas
        relative to the sacrum in the ROTATED coordinate system (Z-axis aligned with gait).
        """
        all_forward_movement = {}
        all_extrema_data = {}
        event_times_data = {}

        event_definitions = {
             "HS_left": ("left_foot_index", "peak"),
             "HS_right": ("right_foot_index", "peak"),
             "TO_left": ("left_foot_index", "valley"),
             "TO_right": ("right_foot_index", "valley")
        }

        # Define filter parameters (get from config or use defaults)
        filter_config = self.config.get("event_detection", {}).get("filter", {})
        cutoff = filter_config.get("cutoff", 3.0)
        order = filter_config.get("order", 4)
        apply_filter = filter_config.get("apply", True) # Add flag to enable/disable filtering

        for event_name, (landmark, extrema_type) in event_definitions.items():
            logger.debug(f"Processing event: {event_name} using landmark: {landmark}")
            try:
                # Check if columns exist before accessing
                landmark_col = (landmark, 'z')
                sacrum_col = ('sacrum', 'z')
                if landmark_col not in rotated_pose_data.columns or sacrum_col not in rotated_pose_data.columns:
                     logger.warning(f"Missing data for landmark '{landmark}' or 'sacrum' for Zeni calculation. Skipping {event_name}.")
                     continue

                forward_movement = rotated_pose_data[landmark_col] - rotated_pose_data[sacrum_col]
                forward_movement_np = forward_movement.to_numpy()

                if apply_filter:
                    filtered_movement = butter_lowpass_filter(forward_movement_np, cutoff=cutoff, fs=self.frame_rate, order=order)
                    signal_for_extrema = filtered_movement # Use filtered signal
                else:
                    signal_for_extrema = forward_movement_np # Use raw signal

                all_forward_movement[event_name] = signal_for_extrema # Store the signal used for extrema detection

            except KeyError as e:
                logger.error(f"KeyError computing forward movement for {event_name}: {e}")
                continue
            except Exception as e:
                logger.error(f"Error computing forward movement for {event_name}: {e}", exc_info=True)
                continue

            try:
                peaks_idx, valleys_idx = detect_extremas(signal_for_extrema)
                logger.debug(f"{event_name}: Found {len(peaks_idx)} peaks, {len(valleys_idx)} valleys on {'filtered' if apply_filter else 'raw'} signal.")
                all_extrema_data[event_name] = {"peaks_indices": peaks_idx, "valleys_indices": valleys_idx}

            except Exception as e:
                logger.error(f"Error detecting extremas for {event_name}: {e}", exc_info=True)
                peaks_idx, valleys_idx = np.array([]), np.array([])

            if extrema_type == "peak":
                event_indices = peaks_idx
            elif extrema_type == "valley":
                event_indices = valleys_idx
            else:
                event_indices = np.array([])

            event_times = event_indices / self.frame_rate
            event_times_data[event_name] = event_times
            logger.debug(f"Detected {event_name} times (s): {np.round(event_times[:5], 3)}...") # Log first few


        # Create the final events DataFrame
        try:
            max_len = 0
            if event_times_data:
                 non_empty_values = [v for v in event_times_data.values() if isinstance(v, (list, np.ndarray)) and len(v) > 0]
                 if non_empty_values:
                      max_len = max(len(v) for v in non_empty_values)

            events_dict_padded = {}
            for key, values in event_times_data.items():
                 padding = [np.nan] * (max_len - len(values))
                 events_dict_padded[key] = np.concatenate((values, padding))

            events = pd.DataFrame(events_dict_padded)
            standard_cols = ['HS_left', 'TO_left', 'HS_right', 'TO_right']
            cols_ordered = [col for col in standard_cols if col in events.columns] + \
                           [col for col in events.columns if col not in standard_cols]
            events = events[cols_ordered]

            logger.info(f"Created events DataFrame with shape: {events.shape}")
            if "HS_left" in events.columns: logger.debug(f"Detected HS_left events (times in s, head):\n{events['HS_left'].dropna().round(3).head().to_string()}")
            if "HS_right" in events.columns: logger.debug(f"Detected HS_right events (times in s, head):\n{events['HS_right'].dropna().round(3).head().to_string()}")
        except Exception as e:
            logger.error(f"Error creating final events DataFrame: {e}", exc_info=True)
            events = pd.DataFrame(columns=['HS_left', 'TO_left', 'HS_right', 'TO_right'])

        return events

# --- Helper Functions for Rotation (Unchanged from previous version) ---

def compute_framewise_rotation_angles(pose_data, marker="sacrum", window_size=100, step_size=50, frame_rate=30):
    """Computes rotation angle for each frame based on gait direction in sliding windows."""
    logger.debug(f"Computing rotation angles: marker={marker}, window={window_size}, step={step_size}")
    marker_x_col = (marker, 'x')
    marker_z_col = (marker, 'z')

    if not isinstance(pose_data.columns, pd.MultiIndex):
         logger.warning("Pose data columns are not MultiIndex in compute_framewise_rotation_angles. Attempting access anyway.")
         # Adjust column names if needed, or assume ensure_multiindex was called
    elif marker_x_col not in pose_data.columns or marker_z_col not in pose_data.columns:
         raise ValueError(f"Marker '{marker}' with coordinates 'x' and 'z' not found in pose_data for rotation calculation.")

    sliding_angles = determine_gait_direction_sliding_window(pose_data, marker, window_size, step_size)

    if not sliding_angles:
        logger.warning("No sliding window angles computed; returning zero angles for all frames.")
        return np.zeros(len(pose_data))

    centers, window_angles = zip(*sliding_angles)
    centers = np.array(centers)
    window_angles = np.array(window_angles)

    num_frames = len(pose_data)
    frame_indices = np.arange(num_frames)
    framewise_angles = np.interp(frame_indices, centers, window_angles, left=window_angles[0], right=window_angles[-1])
    logger.debug(f"Interpolated framewise angles for {num_frames} frames.")
    return framewise_angles


def determine_gait_direction_sliding_window(pose_data, marker="sacrum", window_size=100, step_size=50):
    """Calculates the dominant direction (angle) in the XZ plane for sliding windows."""
    angles = []
    num_frames = len(pose_data)
    if num_frames < window_size:
        logger.warning(f"Data length ({num_frames}) is smaller than window size ({window_size}). Cannot compute sliding window angles.")
        return []

    marker_x_col = (marker, 'x')
    marker_z_col = (marker, 'z')

    for start in range(0, num_frames - window_size + 1, step_size):
        end = start + window_size
        window_data = pose_data.iloc[start:end]

        try:
            x_coords = window_data[marker_x_col].to_numpy()
            z_coords = window_data[marker_z_col].to_numpy()
        except KeyError:
            logger.warning(f"KeyError extracting coordinates for marker '{marker}' in window {start}-{end}. Skipping window.")
            continue

        # Check for NaNs or constant values
        if np.isnan(x_coords).any() or np.isnan(z_coords).any() or \
           (np.all(x_coords == x_coords[0]) and np.all(z_coords == z_coords[0])):
             logger.debug(f"NaNs or constant position found in window {start}-{end}. Skipping PCA.")
             continue

        positions = np.column_stack((x_coords, z_coords))

        try:
            pca = PCA(n_components=1)
            pca.fit(positions)
            principal_vector = pca.components_[0]
            # Angle relative to Z+ axis towards X+ axis
            angle = np.arctan2(principal_vector[0], principal_vector[1])

        except Exception as e:
            logger.warning(f"PCA failed for window {start}-{end}: {e}. Using angle 0.")
            angle = 0

        window_center_frame = start + window_size // 2
        # We want to rotate BY the negative of this angle to align Z with movement direction
        angles.append((window_center_frame, -angle))

    logger.debug(f"Computed {len(angles)} window angles.")
    return angles


def rotate_pose_data_framewise(pose_data, rotation_angles):
    """Rotates the XZ coordinates of all markers for each frame by the given angle."""
    if len(rotation_angles) != len(pose_data):
        raise ValueError(f"Length of rotation_angles ({len(rotation_angles)}) must match the number of frames in pose_data ({len(pose_data)}).")

    rotated_data = pose_data.copy()
    # Get unique marker names from the first level of the MultiIndex
    markers = pose_data.columns.get_level_values(0).unique()

    cos_a = np.cos(rotation_angles)
    sin_a = np.sin(rotation_angles)

    for marker in markers:
        x_col = (marker, 'x')
        z_col = (marker, 'z')

        # Check if both x and z coordinates exist for the marker using the tuple directly
        if x_col in pose_data.columns and z_col in pose_data.columns:
            try:
                x_orig = pose_data[x_col].to_numpy()
                z_orig = pose_data[z_col].to_numpy()

                # Apply 2D rotation using pre-calculated sin/cos
                new_x = x_orig * cos_a - z_orig * sin_a
                new_z = x_orig * sin_a + z_orig * cos_a

                rotated_data[x_col] = new_x
                rotated_data[z_col] = new_z
            except Exception as e:
                logger.warning(f"Error rotating marker '{marker}': {e}") # Log as warning

    return rotated_data