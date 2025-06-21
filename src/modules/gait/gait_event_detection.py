# --- START OF FILE gait_event_detection.py ---

import numpy as np
import pandas as pd
import logging
from sklearn.decomposition import PCA
import os
import json

# Import the MODIFIED detect_extremas and filtering function
from my_utils.helpers import detect_extremas, butter_lowpass_filter
# Import only the needed plot function (plot generation is now handled in gait_pipeline.py)
# from modules.gait.my_utils.plotting import plot_combined_extremas_and_toe # Plotting done elsewhere

# Configure logger for this module
logger = logging.getLogger(__name__)

class EventDetector:
    """
    Detects heel strike (HS) and toe-off (TO) events from pose data using rotated coordinates.
    Modified to return both events DataFrame and the rotated pose data.
    Uses prominence and distance for extrema detection.
    """

    def __init__(self, input_path, algorithm="zeni", frame_rate=30.0, window_size=100, step_size=50, config=None, **kwargs):
        """
        Initializes the EventDetector.

        Args:
            input_path (str): Path to the input file (for context, not always used directly).
            algorithm (str): Event detection algorithm ('zeni').
            frame_rate (float): Frame rate of the video/data.
            window_size (int): Window size (frames) for PCA rotation.
            step_size (int): Step size (frames) for PCA rotation.
            config (dict): Full configuration dictionary for accessing nested parameters.
            **kwargs: Additional keyword arguments (currently unused).
        """
        self.input_path = input_path
        self.algorithm = algorithm
        self.frame_rate = float(frame_rate) # Ensure frame rate is float
        # Ensure window/step size are reasonable positive integers
        self.window_size = max(10, int(window_size))
        self.step_size = max(1, int(step_size))
        self.config = config if config is not None else {} # Use provided config or empty dict

        if self.frame_rate <= 0:
             logger.error(f"Invalid frame_rate ({self.frame_rate}) passed to EventDetector.")
             raise ValueError("Frame rate must be positive.")

        logger.info(f"EventDetector initialized with algorithm='{self.algorithm}', frame_rate={self.frame_rate:.2f} fps, rotation_window_size={self.window_size} frames, rotation_step_size={self.step_size} frames.")


    def detect_heel_toe_events(self, pose_data):
        """
        Detects gait events and returns both the events DataFrame and the rotated pose data used.
        """
        logger.debug(f"Pose data columns received in detect_heel_toe_events: {pose_data.columns.tolist()}")

        # Ensure MultiIndex for consistency
        if not isinstance(pose_data.columns, pd.MultiIndex):
            from modules.gait.gait_parameters_computation import ensure_multiindex
            pose_data = ensure_multiindex(pose_data)
            logger.debug("Converted pose_data columns to MultiIndex.")

        # Define required markers for rotation and event detection
        required_markers_rotation = ['sacrum']
        required_markers_events = ['left_foot_index', 'right_foot_index', 'sacrum'] # Zeni uses foot_index and sacrum

        # Check marker presence
        present_markers = set(col[0] for col in pose_data.columns if isinstance(col, tuple))
        missing_rotation = [m for m in required_markers_rotation if m not in present_markers]
        missing_events = [m for m in required_markers_events if m not in present_markers]

        if missing_rotation:
            logger.error(f"Missing required markers in pose_data for rotation: {missing_rotation}")
            raise ValueError(f"Pose data missing required markers for rotation: {missing_rotation}")
        if missing_events:
             logger.error(f"Missing required markers in pose_data for Zeni event detection: {missing_events}")
             raise ValueError(f"Pose data missing required markers for event detection: {missing_events}")


        # --- Rotation ---
        rotated_pose_data = None
        try:
            logger.info("Computing framewise rotation angles...")
            # Use specific rotation marker from config if available, else default to sacrum
            rotation_marker = self.config.get("event_detection", {}).get("rotation_marker", "sacrum")
            rotation_angles = compute_framewise_rotation_angles(
                pose_data,
                marker=rotation_marker,
                window_size=self.window_size,
                step_size=self.step_size,
                frame_rate=self.frame_rate # Pass frame rate if needed by helper
            )
            logger.info("Finished computing rotation angles.")

            logger.info("Rotating pose data framewise...")
            rotated_pose_data = rotate_pose_data_framewise(pose_data, rotation_angles)
            logger.info("Finished rotating pose data.")

        except Exception as e:
            logger.error(f"Error during pose data rotation: {e}", exc_info=True)
            # Decide how to handle rotation failure: Stop? Or proceed with original data?
            # Let's proceed with original data but log a clear warning.
            logger.warning("Proceeding with event detection using ORIGINAL (non-rotated) pose data due to rotation error.")
            rotated_pose_data = pose_data # Use original data as fallback


        # --- Event Detection ---
        events = pd.DataFrame() # Initialize empty DataFrame
        try:
            logger.info(f"Detecting events using algorithm: {self.algorithm}")
            if self.algorithm == "zeni":
                # Pass the (potentially original) data to the Zeni method
                events = self._detect_events_zeni(rotated_pose_data)
            else:
                logger.error(f"Unsupported event detection algorithm: {self.algorithm}")
                # Return empty events and the data used (rotated or original)
                return events, rotated_pose_data

            if events.empty:
                 logger.warning("Event detection algorithm finished but returned no events.")
            else:
                 logger.info(f"Finished event detection. Found {events.notna().sum().sum()} total event timings.") # Log total non-NaN events

        except Exception as e:
            logger.error(f"Error during event detection using algorithm {self.algorithm}: {e}", exc_info=True)
            # Return empty events but the data used
            return events, rotated_pose_data

        # *** RETURN BOTH events AND the data used for detection ***
        return events, rotated_pose_data


    def _detect_events_zeni(self, pose_data_for_detection): # Argument is the data (rotated or original) to use
        """
        Detects events using the Zeni method based on forward displacement extremas
        relative to the sacrum in the provided coordinate system (ideally rotated).
        Uses prominence and distance for peak/valley detection.
        """
        all_forward_movement = {}
        all_extrema_data = {}
        event_times_data = {} # Stores lists of event times in seconds

        # Zeni uses foot_index relative to sacrum Z coordinate
        event_definitions = {
             "HS_left": ("left_foot_index", "peak"),   # Peak forward displacement = HS
             "HS_right": ("right_foot_index", "peak"),  # Peak forward displacement = HS
             "TO_left": ("left_foot_index", "valley"), # Valley forward displacement = TO
             "TO_right": ("right_foot_index", "valley") # Valley forward displacement = TO
        }

        # --- Get Configuration Settings ---
        event_config = self.config.get("event_detection", {})
        filter_config = event_config.get("filter", {})
        extrema_config = event_config.get("extrema_detection", {})

        cutoff = filter_config.get("cutoff", 3.0)
        order = filter_config.get("order", 4)
        apply_filter = filter_config.get("apply", True)

        # Get prominence/distance from config for detect_extremas
        prominence = extrema_config.get("prominence", 0.1) # Default 0.1 if not in config
        min_frames_between = extrema_config.get("min_frames_between_events", None) # Default None
        # Convert distance to int if provided
        distance = int(min_frames_between) if min_frames_between is not None and min_frames_between > 0 else None
        # ------------------------------------

        # Define columns based on Zeni method (foot relative to sacrum in Z)
        sacrum_col = ('sacrum', 'z')
        if sacrum_col not in pose_data_for_detection.columns:
             logger.error(f"Sacrum Z-coordinate '{sacrum_col}' not found in input data. Cannot perform Zeni detection.")
             return pd.DataFrame() # Return empty if sacrum is missing

        # --- Process Each Event Type ---
        for event_name, (landmark, extrema_type) in event_definitions.items():
            logger.debug(f"Processing event: {event_name} using landmark: {landmark}")
            landmark_col = (landmark, 'z')

            if landmark_col not in pose_data_for_detection.columns:
                 logger.warning(f"Landmark Z-coordinate '{landmark_col}' not found. Skipping event {event_name}.")
                 event_times_data[event_name] = np.array([]) # Ensure key exists even if skipped
                 continue

            try:
                # Calculate forward movement signal (Foot Z - Sacrum Z)
                forward_movement = pose_data_for_detection[landmark_col] - pose_data_for_detection[sacrum_col]
                forward_movement_np = forward_movement.to_numpy()

                # Handle potential NaNs in signal before filtering/detection
                if np.isnan(forward_movement_np).all():
                     logger.warning(f"Forward movement signal for {event_name} contains only NaNs. Skipping.")
                     event_times_data[event_name] = np.array([])
                     continue
                # Optional: Interpolate NaNs if filtering is sensitive
                # nan_mask = np.isnan(forward_movement_np)
                # if nan_mask.any():
                #     indices = np.arange(len(forward_movement_np))
                #     forward_movement_np[nan_mask] = np.interp(indices[nan_mask], indices[~nan_mask], forward_movement_np[~nan_mask])
                #     logger.debug(f"Interpolated NaNs in forward movement signal for {event_name}.")

                # Apply filter if configured
                if apply_filter:
                    # Check signal length before filtering
                    if len(forward_movement_np) > order * 3:
                        filtered_movement = butter_lowpass_filter(forward_movement_np, cutoff=cutoff, fs=self.frame_rate, order=order)
                        signal_for_extrema = filtered_movement # Use filtered signal
                        logger.debug(f"Applied Butterworth filter to {event_name} signal (cutoff={cutoff}, order={order}).")
                    else:
                        logger.warning(f"Signal length too short for filtering ({len(forward_movement_np)} <= {order*3}). Using raw signal for {event_name}.")
                        signal_for_extrema = forward_movement_np # Use raw signal if too short
                else:
                    signal_for_extrema = forward_movement_np # Use raw signal
                    logger.debug(f"Using raw (unfiltered) signal for {event_name}.")

                all_forward_movement[event_name] = signal_for_extrema # Store the signal used

            except Exception as e:
                logger.error(f"Error processing signal for {event_name}: {e}", exc_info=True)
                event_times_data[event_name] = np.array([])
                continue

            # --- Detect Extremas ---
            try:
                # Pass prominence and distance to the updated detect_extremas function
                peaks_idx, valleys_idx = detect_extremas(
                    signal_for_extrema,
                    prominence=prominence,
                    distance=distance
                )
                logger.debug(f"{event_name}: Found {len(peaks_idx)} peaks, {len(valleys_idx)} valleys using prominence={prominence}, distance={distance}.")
                # Store indices for potential later use/debugging
                all_extrema_data[event_name] = {"peaks_indices": peaks_idx, "valleys_indices": valleys_idx}

            except Exception as e:
                logger.error(f"Error detecting extremas for {event_name}: {e}", exc_info=True)
                peaks_idx, valleys_idx = np.array([]), np.array([]) # Ensure empty arrays on error

            # Select the correct type of extrema based on event definition
            if extrema_type == "peak":
                event_indices = peaks_idx
            elif extrema_type == "valley":
                event_indices = valleys_idx
            else:
                logger.warning(f"Unknown extrema type '{extrema_type}' defined for {event_name}.")
                event_indices = np.array([])

            # Convert indices to times in seconds
            event_times = event_indices / self.frame_rate
            event_times_data[event_name] = event_times
            logger.debug(f"Detected {event_name} times (s) - Count: {len(event_times)}, First 5: {np.round(event_times[:5], 3)}")


        # --- Create the final events DataFrame ---
        try:
            # Determine the maximum number of events found for any type
            max_len = 0
            if event_times_data:
                 non_empty_values = [v for v in event_times_data.values() if isinstance(v, np.ndarray) and v.size > 0]
                 if non_empty_values:
                      max_len = max(len(v) for v in non_empty_values)

            events_dict_padded = {}
            standard_cols = ['HS_left', 'TO_left', 'HS_right', 'TO_right']

            # Ensure all standard columns exist in the dict, even if empty
            for col in standard_cols:
                 if col not in event_times_data:
                      event_times_data[col] = np.array([])

            # Pad each event list with NaNs to the max length
            for key, values in event_times_data.items():
                 if not isinstance(values, np.ndarray): # Ensure it's an array
                      values = np.array(values)
                 padding = np.full(max_len - len(values), np.nan) # Use np.full for clarity
                 events_dict_padded[key] = np.concatenate((values, padding))

            # Create DataFrame and order columns
            events = pd.DataFrame(events_dict_padded)
            cols_ordered = [col for col in standard_cols if col in events.columns] + \
                           [col for col in events.columns if col not in standard_cols] # Add any non-standard ones
            events = events[cols_ordered]

            logger.info(f"Created final events DataFrame with shape: {events.shape}")
            if not events.empty:
                 # Log counts of non-NaN events detected per type
                 logger.debug(f"Detected event counts:\n{events.notna().sum().to_string()}")

        except Exception as e:
            logger.error(f"Error creating final events DataFrame: {e}", exc_info=True)
            events = pd.DataFrame(columns=standard_cols) # Return empty DF with standard columns on error

        return events


# --- Helper Functions for Rotation (Should remain unchanged if they work) ---

def compute_framewise_rotation_angles(pose_data, marker="sacrum", window_size=100, step_size=50, frame_rate=30):
    """Computes rotation angle for each frame based on gait direction in sliding windows."""
    logger.debug(f"Computing rotation angles: marker={marker}, window={window_size}, step={step_size}")
    marker_x_col = (marker, 'x')
    marker_z_col = (marker, 'z')

    if not isinstance(pose_data.columns, pd.MultiIndex):
         logger.warning("Pose data columns are not MultiIndex in compute_framewise_rotation_angles. Attempting access anyway.")
         # Assuming column names might be like 'sacrum_x', 'sacrum_z'
         marker_x_col = f"{marker}_x"
         marker_z_col = f"{marker}_z"
         if marker_x_col not in pose_data.columns or marker_z_col not in pose_data.columns:
              raise ValueError(f"Marker '{marker}' flat columns '{marker_x_col}'/'{marker_z_col}' not found.")
    elif marker_x_col not in pose_data.columns or marker_z_col not in pose_data.columns:
         raise ValueError(f"Marker '{marker}' MultiIndex columns '{marker_x_col}'/'{marker_z_col}' not found.")

    # Determine angles using sliding window PCA
    sliding_angles = determine_gait_direction_sliding_window(pose_data, marker, window_size, step_size)

    if not sliding_angles:
        logger.warning("No sliding window angles computed; returning zero angles for all frames.")
        return np.zeros(len(pose_data))

    # Unpack centers (frame indices) and window angles
    centers, window_angles = zip(*sliding_angles)
    centers = np.array(centers)
    window_angles = np.array(window_angles) # These are the angles TO ROTATE BY

    # Interpolate to get an angle for every frame
    num_frames = len(pose_data)
    frame_indices = np.arange(num_frames)
    # Use window_angles[0] for frames before the first center, window_angles[-1] for frames after the last center
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

    # Handle potential flat index from warning in caller
    if not isinstance(pose_data.columns, pd.MultiIndex):
        marker_x_col = f"{marker}_x"
        marker_z_col = f"{marker}_z"
    else:
        marker_x_col = (marker, 'x')
        marker_z_col = (marker, 'z')

    for start in range(0, num_frames - window_size + 1, step_size):
        end = start + window_size
        window_data = pose_data.iloc[start:end]

        try:
            # Extract coordinates, handle potential NaNs
            x_coords = window_data[marker_x_col].dropna().to_numpy()
            z_coords = window_data[marker_z_col].dropna().to_numpy()

            # Ensure we have enough points after dropping NaNs
            if len(x_coords) < 2 or len(z_coords) < 2 or len(x_coords) != len(z_coords):
                 logger.debug(f"Not enough valid (non-NaN) data points in window {start}-{end}. Skipping PCA.")
                 continue

        except KeyError:
            logger.warning(f"KeyError extracting coordinates for marker '{marker}' in window {start}-{end}. Skipping window.")
            continue

        # Check for constant values (no movement)
        if (np.all(x_coords == x_coords[0]) and np.all(z_coords == z_coords[0])):
             logger.debug(f"Constant position found in window {start}-{end}. Skipping PCA.")
             angle_to_rotate_by = 0 # Assign a default angle (e.g., 0) if no movement
        else:
            positions = np.column_stack((x_coords, z_coords))
            try:
                pca = PCA(n_components=1)
                pca.fit(positions)
                # principal_vector[0] is movement along X, principal_vector[1] is movement along Z
                principal_vector = pca.components_[0]
                # Angle of the principal component vector relative to Z+ axis (towards X+)
                movement_angle = np.arctan2(principal_vector[0], principal_vector[1])
                # We want to rotate BY the negative of this angle to align Z with movement
                angle_to_rotate_by = -movement_angle

            except Exception as e:
                logger.warning(f"PCA failed for window {start}-{end}: {e}. Using angle 0.")
                angle_to_rotate_by = 0

        window_center_frame = start + window_size // 2
        angles.append((window_center_frame, angle_to_rotate_by)) # Store frame index and angle TO ROTATE BY

    logger.debug(f"Computed {len(angles)} window angles.")
    return angles


def rotate_pose_data_framewise(pose_data, rotation_angles):
    """Rotates the XZ coordinates of all markers for each frame by the given angle."""
    if len(rotation_angles) != len(pose_data):
        raise ValueError(f"Length of rotation_angles ({len(rotation_angles)}) must match the number of frames in pose_data ({len(pose_data)}).")

    rotated_data = pose_data.copy()
    # Get unique marker names (first level of MultiIndex)
    markers = pose_data.columns.get_level_values(0).unique()

    # Pre-calculate sin and cos for all angles
    cos_a = np.cos(rotation_angles)
    sin_a = np.sin(rotation_angles)

    for marker in markers:
        x_col = (marker, 'x')
        z_col = (marker, 'z')

        # Check if both x and z coordinates exist for the marker
        if x_col in pose_data.columns and z_col in pose_data.columns:
            try:
                x_orig = pose_data[x_col].to_numpy()
                z_orig = pose_data[z_col].to_numpy()

                # Apply 2D rotation using NumPy broadcasting (efficient for large arrays)
                # new_x = x' = x*cos(a) - z*sin(a)
                # new_z = z' = x*sin(a) + z*cos(a)
                new_x = x_orig * cos_a - z_orig * sin_a
                new_z = x_orig * sin_a + z_orig * cos_a

                # Assign rotated values back to the DataFrame
                rotated_data[x_col] = new_x
                rotated_data[z_col] = new_z
            except Exception as e:
                # Log warning but continue with other markers
                logger.warning(f"Error rotating marker '{marker}': {e}")

    return rotated_data

# --- END OF FILE gait_event_detection.py ---