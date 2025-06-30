import numpy as np
import pandas as pd

from scipy.signal import medfilt

# Corrected import:
# Assuming my_utils is a package directly under src, and this file (gait_preprocessing.py)
# is in src/gait/.
# Python will look in sys.path. If src/ is effectively on sys.path (due to how script is run),
# then 'my_utils' can be found directly.
from my_utils.helpers import log, butter_lowpass_filter

class Preprocessor:
    # ... (rest of your Preprocessor class code is fine) ...
    def __init__(self, pose_data):
        """
        Initializes the Preprocessor with pose data.

        Parameters:
            pose_data (DataFrame): MultiIndex DataFrame containing gait coordinates (x, y, z) for various landmarks.
        """
        if not isinstance(pose_data, pd.DataFrame):
            raise ValueError("pose_data must be a pandas DataFrame")
        self.pose_data = pose_data
        # Make sure 'log' is defined if you use it here, or pass a logger
        # For simplicity, assuming 'log' from helpers is intended to be used.
        log("Preprocessor initialized with pose data.", level="INFO")

    def compute_sacrum(self):
        """
        Computes the sacrum position as the midpoint between the left and right hips.
        """
        
        def compute_midpoint(p1, p2, axis):
            """Computes the midpoint between two points along a given axis."""
            return (self.pose_data[(p1, axis)] + self.pose_data[(p2, axis)]) / 2

        try:
            for axis in ['x', 'y', 'z']:
                self.pose_data[('sacrum', axis)] = compute_midpoint('left_hip', 'right_hip', axis)                 
        except KeyError as e:
            log(f"Missing columns for sacrum calculation: {e}", level="ERROR")
            # Consider re-raising or handling more gracefully depending on how critical sacrum is
            raise KeyError(f"Missing expected columns in pose_data for sacrum calculation: {e}")
           
    def handle_missing_values(self):
        """
        Interpolate missing values using linear interpolation.
        """
        # Ensure operating on numeric columns only for interpolation
        numeric_cols = self.pose_data.select_dtypes(include=np.number).columns
        self.pose_data[numeric_cols] = self.pose_data[numeric_cols].interpolate(method='linear', limit_direction='both')
        # limit_direction='both' helps fill NaNs at the beginning/end if possible
        # An alternative or addition is ffill/bfill if some remain
        # self.pose_data[numeric_cols] = self.pose_data[numeric_cols].ffill().bfill() 
        log("Missing values handled by interpolation.", level="DEBUG") # Changed to DEBUG
    
    def normalize(self, window_size=31):
        """
        Normalizes pose data by the distance between specific landmarks (e.g., knee and ankle).
        """
        log("Starting normalization process...", level="INFO")

        required_landmarks_coords = [
            ('left_knee', 'x'), ('left_knee', 'y'), ('left_knee', 'z'),
            ('left_ankle', 'x'), ('left_ankle', 'y'), ('left_ankle', 'z'),
            ('right_knee', 'x'), ('right_knee', 'y'), ('right_knee', 'z'),
            ('right_ankle', 'x'), ('right_ankle', 'y'), ('right_ankle', 'z')
        ]
        missing_for_norm = [lc for lc in required_landmarks_coords if lc not in self.pose_data.columns]
        if missing_for_norm:
            msg = f"Missing columns for leg length calculation during normalization: {missing_for_norm}. Skipping normalization."
            log(msg, level="WARNING")
            # raise KeyError(msg) # Or just skip normalization
            return # Skip normalization if essential columns are missing

        try:
            left_leg_length = np.sqrt(
                (self.pose_data[('left_knee', 'x')] - self.pose_data[('left_ankle', 'x')])**2 +
                (self.pose_data[('left_knee', 'y')] - self.pose_data[('left_ankle', 'y')])**2 +
                (self.pose_data[('left_knee', 'z')] - self.pose_data[('left_ankle', 'z')])**2
            )
            right_leg_length = np.sqrt(
                (self.pose_data[('right_knee', 'x')] - self.pose_data[('right_ankle', 'x')])**2 +
                (self.pose_data[('right_knee', 'y')] - self.pose_data[('right_ankle', 'y')])**2 +
                (self.pose_data[('right_knee', 'z')] - self.pose_data[('right_ankle', 'z')])**2
            )
        except KeyError as e: # Should be caught by the check above, but as a fallback
            log(f"Unexpected KeyError during leg length calculation: {e}. Skipping normalization.", level="ERROR")
            return

        # Ensure window_size is odd and appropriate for medfilt
        if window_size <= 0 : window_size = 1 # Must be positive
        if window_size % 2 == 0: window_size += 1 # Must be odd

        # Apply median filter properly to Pandas Series then convert to NumPy array if medfilt needs it
        # or ensure medfilt can handle Series. scipy.signal.medfilt expects array-like.
        left_leg_length_np = left_leg_length.to_numpy()
        right_leg_length_np = right_leg_length.to_numpy()

        # Check if length of array is sufficient for kernel size
        if len(left_leg_length_np) >= window_size:
            left_leg_length_filtered = medfilt(left_leg_length_np, kernel_size=window_size)
        else:
            log(f"Data length ({len(left_leg_length_np)}) too short for median filter window ({window_size}). Using unfiltered left leg length.", level="WARNING")
            left_leg_length_filtered = left_leg_length_np
        
        if len(right_leg_length_np) >= window_size:
            right_leg_length_filtered = medfilt(right_leg_length_np, kernel_size=window_size)
        else:
            log(f"Data length ({len(right_leg_length_np)}) too short for median filter window ({window_size}). Using unfiltered right leg length.", level="WARNING")
            right_leg_length_filtered = right_leg_length_np


        # Add small epsilon to avoid division by zero
        left_leg_length_filtered = np.where(left_leg_length_filtered == 0, 1e-6, left_leg_length_filtered) + 1e-6
        right_leg_length_filtered = np.where(right_leg_length_filtered == 0, 1e-6, right_leg_length_filtered) + 1e-6


        # Normalize each landmark's coordinates
        # Create a copy for modification if self.pose_data should not be modified in place by this method
        # normalized_pose_data = self.pose_data.copy() # If not modifying in-place
        for landmark in set(self.pose_data.columns.get_level_values(0)):
            for coord in ['x', 'y', 'z']:
                if (landmark, coord) in self.pose_data.columns:
                    # Ensure the landmark is part of a leg if specific leg length is used
                    if 'left' in str(landmark).lower(): # Check if 'left' is in the landmark name
                        self.pose_data.loc[:, (landmark, coord)] /= left_leg_length_filtered
                    elif 'right' in str(landmark).lower(): # Check if 'right' is in the landmark name
                        self.pose_data.loc[:, (landmark, coord)] /= right_leg_length_filtered
                    # else: # For non-leg landmarks, decide on normalization strategy (e.g., average leg length, or skip)
                        # For simplicity, only normalizing leg/foot landmarks here.
                        # If you want to normalize torso too, use an average or a different reference.

        log("Normalization complete.", level="INFO")
        # return normalized_pose_data # If not modifying in-place

    def preprocess(self, window_size):
        """
        Executes the full preprocessing pipeline: sacrum computation, missing value handling,
        Butterworth filtering, and normalization. Modifies self.pose_data in place.

        Returns:
            DataFrame: Fully preprocessed pose data (self.pose_data).
        """
        log("Starting full preprocessing pipeline...", level="INFO")
        
        self.compute_sacrum() # Modifies self.pose_data
        log("Sacrum computation complete.", level="DEBUG")

        self.handle_missing_values() # Modifies self.pose_data
        log("Missing value handling complete.", level="DEBUG")

        # butter_lowpass_filter returns a new DataFrame. Reassign to self.pose_data.
        # Ensure butter_lowpass_filter from helpers.py is configured (fs, cutoff from config.json ideally)
        # For now, assuming Preprocessor gets pose_data and applies a default or configured filter.
        # This part might need access to filtering parameters from config.json
        # Let's assume for now it uses some defaults or gets them from elsewhere if needed.
        # If butter_lowpass_filter is in helpers and not configured here, it might use its own defaults.
        filter_config = {} # Placeholder: ideally, this comes from main config
        fs = filter_config.get("fs", 30) # Example, should come from actual data FPS
        cutoff = filter_config.get("cutoff", 6.0) # Example default from some biomechanics literature
        order = filter_config.get("order", 4)

        self.pose_data = butter_lowpass_filter(self.pose_data, cutoff=cutoff, fs=fs, order=order)
        log(f"Butterworth lowpass filtering complete (fs={fs}, cutoff={cutoff}, order={order}).", level="DEBUG")
        
        # Normalize method modifies self.pose_data in place based on current implementation.
        # If normalize returns a DataFrame, then self.pose_data = self.normalize(...)
        self.normalize(window_size=window_size) 
        log("Normalization step complete.", level="DEBUG")
        
        log("Preprocessing pipeline complete.", level="INFO")
        return self.pose_data