import numpy as np
import pandas as pd

from scipy.signal import medfilt

from modules.gait.my_utils.helpers import log, butter_lowpass_filter

class Preprocessor:
    def __init__(self, pose_data):
        """
        Initializes the Preprocessor with pose data.

        Parameters:
            pose_data (DataFrame): MultiIndex DataFrame containing gait coordinates (x, y, z) for various landmarks.
        """
        if not isinstance(pose_data, pd.DataFrame):
            raise ValueError("pose_data must be a pandas DataFrame")
        self.pose_data = pose_data
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
            raise KeyError(f"Missing expected columns in pose_data: {e}")
           
    def handle_missing_values(self):
        """
        Interpolate missing values using linear interpolation.
        """
        numeric_columns = self.pose_data.select_dtypes(include=['number']).columns
        self.pose_data[numeric_columns] = (
            self.pose_data[numeric_columns].interpolate(method='linear').ffill().bfill()
        )
    
    def normalize(self, window_size=31):
        """
        Normalizes pose data by the distance between specific landmarks (e.g., knee and ankle).

        Parameters:
            window_size (int): Size of the window for the median filter. Must be an odd number.

        Returns:
            DataFrame: Normalized pose data.
        """
        log("Starting normalization process...", level="INFO")

        try:
            left_leg_length = np.sqrt(
                (self.pose_data[('left_knee', 'x')] - self.pose_data[('left_ankle', 'x')]) ** 2 +
                (self.pose_data[('left_knee', 'y')] - self.pose_data[('left_ankle', 'y')]) ** 2 +
                (self.pose_data[('left_knee', 'z')] - self.pose_data[('left_ankle', 'z')]) ** 2
            )
            right_leg_length = np.sqrt(
                (self.pose_data[('right_knee', 'x')] - self.pose_data[('right_ankle', 'x')]) ** 2 +
                (self.pose_data[('right_knee', 'y')] - self.pose_data[('right_ankle', 'y')]) ** 2 +
                (self.pose_data[('right_knee', 'z')] - self.pose_data[('right_ankle', 'z')]) ** 2
            )
        except KeyError as e:
            log(f"Missing columns for leg length calculation: {e}", level="ERROR")
            raise KeyError(f"Missing expected columns in filtered_pose_data: {e}")

        # Apply median filter properly
        left_leg_length = medfilt(left_leg_length, kernel_size=window_size)
        right_leg_length = medfilt(right_leg_length, kernel_size=window_size)

        # Add small epsilon to avoid division by zero
        left_leg_length += 1e-6
        right_leg_length += 1e-6

        # Normalize each landmark's coordinates
        for landmark in set(self.pose_data.columns.get_level_values(0)):  # Fix MultiIndex handling
            for coord in ['x', 'y', 'z']:
                if (landmark, coord) in self.pose_data.columns:
                    if 'left' in landmark:
                        self.pose_data[(landmark, coord)] /= left_leg_length
                    elif 'right' in landmark:
                        self.pose_data[(landmark, coord)] /= right_leg_length

        log("Normalization complete.", level="INFO")

    def preprocess(self, window_size):
            """
            Executes the full preprocessing pipeline: filtering and normalization.

            Returns:
                DataFrame: Fully preprocessed pose data.
            """
            log("Starting full preprocessing pipeline...", level="INFO")
            self.compute_sacrum()
            self.handle_missing_values()
            self.pose_data = butter_lowpass_filter(self.pose_data)
            self.normalize(window_size=window_size)
            log("Preprocessing pipeline complete.", level="INFO")
            return self.pose_data


