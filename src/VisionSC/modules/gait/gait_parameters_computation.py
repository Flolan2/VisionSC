import os
import json
import numpy as np
import pandas as pd
import logging

from modules.gait.my_utils.gait_parameters import prepare_gait_dataframe
from modules.gait.my_utils.helpers import save_csv

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def ensure_multiindex(pose_data):
    """
    If pose_data has flat column names, convert them to a MultiIndex by splitting on the last underscore.
    For example, "left_foot_index_x" becomes ("left_foot_index", "x").
    """
    if not isinstance(pose_data.columns, pd.MultiIndex):
        new_columns = []
        for col in pose_data.columns:
            if '_' in col:
                parts = col.rsplit('_', 1)
                new_columns.append((parts[0], parts[1]))
            else:
                new_columns.append((col, ''))
        pose_data.columns = pd.MultiIndex.from_tuples(new_columns)
    return pose_data


class GaitParameters:
    @staticmethod
    def compute_parameters(events, pose_data, frame_rate, save_path=None):
        """
        Computes gait parameters for left and right sides and returns a DataFrame with results.
        If a save_path is provided (or determined from the configuration), the results are also saved as CSV.
        """
        if isinstance(events, dict):
            events = pd.DataFrame(events)  # Convert dict to DataFrame if needed

        # Ensure pose_data columns are MultiIndex
        pose_data = ensure_multiindex(pose_data)

        # If no save_path is provided, try to load the default from config.json
        if save_path is None:
            config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../config.json"))
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
                default_dir = config.get("gait_parameters", {}).get("save_path", "../../output/gait_parameters")
                default_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), default_dir))
                os.makedirs(default_dir, exist_ok=True)
                save_path = os.path.join(default_dir, "gait_parameters.csv")
            except Exception as e:
                logger.warning(f"Could not load config for gait_parameters output path: {e}")
                save_path = None

        gait_df = prepare_gait_dataframe()

        # Compute stride duration for each side as the difference between consecutive heel strikes on the same side.
        for side in ['left', 'right']:
            try:
                hs_times = events[f'HS_{side}'].dropna().values
                if len(hs_times) > 1:
                    stride_durations = np.diff(hs_times)
                    gait_df[(side, 'stride_duration')] = pd.Series(stride_durations)
                else:
                    gait_df[(side, 'stride_duration')] = pd.Series()
            except Exception as e:
                logger.exception(f"Error computing stride_duration for {side}: {e}")

        # Compute robust step durations using merged left and right heel strikes.
        try:
            left_steps, right_steps = GaitParameters.compute_robust_step_durations(events)
            gait_df[('left', 'step_duration')] = pd.Series(left_steps)
            gait_df[('right', 'step_duration')] = pd.Series(right_steps)
        except Exception as e:
            logger.exception(f"Error computing robust step_duration: {e}")

        # Compute cadence as 60 / step_duration (replace 0 with NaN to avoid division by zero)
        for side in ['left', 'right']:
            try:
                step_duration_series = gait_df[(side, 'step_duration')].replace(0, np.nan)
                gait_df[(side, 'cadence')] = 60 / step_duration_series
            except Exception as e:
                logger.exception(f"Error computing cadence for {side}: {e}")

        # Compute step length for each side using the provided method.
        for side in ['left', 'right']:
            other_side = 'left' if side == 'right' else 'right'
            try:
                gait_df[(side, 'step_length')] = GaitParameters.compute_step_length(
                    events, pose_data, frame_rate, side, other_side
                )
            except Exception as e:
                logger.exception(f"Error computing step_length for {side}: {e}")

        # Compute stride length as the sum of two consecutive step lengths.
        for side in ['left', 'right']:
            try:
                gait_df[(side, 'stride_length')] = gait_df[(side, 'step_length')].shift(-1) + gait_df[(side, 'step_length')]
            except Exception as e:
                logger.exception(f"Error computing stride_length for {side}: {e}")

        # Compute gait speed as stride_length / stride_duration
        for side in ['left', 'right']:
            try:
                gait_df[(side, 'gait_speed')] = gait_df[(side, 'stride_length')] / gait_df[(side, 'stride_duration')]
            except Exception as e:
                logger.exception(f"Error computing gait_speed for {side}: {e}")

        # Compute swing time as the difference between the next heel strike and toe-off for the same side.
        for side in ['left', 'right']:
            try:
                gait_df[(side, 'swing')] = events[f'HS_{side}'].shift(-1) - events[f'TO_{side}']
            except Exception as e:
                logger.exception(f"Error computing swing for {side}: {e}")

        # Compute initial double support as the difference between the toe-off of the opposite side and the heel strike of the current side.
        for side in ['left', 'right']:
            other_side = 'left' if side == 'right' else 'right'
            try:
                gait_df[(side, 'initial_double_support')] = events[f'TO_{other_side}'] - events[f'HS_{side}']
            except Exception as e:
                logger.exception(f"Error computing initial_double_support for {side}: {e}")

        # Asymmetry calculations for selected features.
        for feature in ['stride_duration', 'cadence', 'swing', 'initial_double_support']:
            try:
                gait_df[('asymmetry', feature)] = (
                    abs(gait_df[('left', feature)] - gait_df[('right', feature)])
                    / gait_df[[('left', feature), ('right', feature)]].max(axis=1) * 100
                )
            except Exception as e:
                logger.exception(f"Error computing asymmetry for feature {feature}: {e}")

        # Save CSV if a valid save_path is determined.
        if save_path:
            try:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                save_csv(gait_df, save_path)
            except Exception as e:
                logger.exception(f"Error saving gait parameters CSV: {e}")

        return gait_df

    @staticmethod
    def compute_robust_step_durations(events, min_valid_duration=0.3):
        """
        Computes robust step durations by merging left and right heel strikes
        and pairing them based on the actual sequence of events.
        Returns two lists: left_step_durations and right_step_durations.
        
        Any paired duration below min_valid_duration (in seconds) is ignored.
        """
        try:
            left_times = events['HS_left'].dropna().values
            right_times = events['HS_right'].dropna().values

            left_df = pd.DataFrame({'time': left_times, 'side': 'left'})
            right_df = pd.DataFrame({'time': right_times, 'side': 'right'})
            combined = pd.concat([left_df, right_df]).sort_values('time').reset_index(drop=True)

            left_step_durations = []
            right_step_durations = []
            for i in range(len(combined) - 1):
                current = combined.iloc[i]
                nxt = combined.iloc[i + 1]
                duration = nxt['time'] - current['time']
                # Only consider durations that are physiologically plausible.
                if duration < min_valid_duration:
                    continue
                if current['side'] == 'left' and nxt['side'] == 'right':
                    left_step_durations.append(duration)
                elif current['side'] == 'right' and nxt['side'] == 'left':
                    right_step_durations.append(duration)
            return left_step_durations, right_step_durations
        except Exception as e:
            logger.exception(f"Error computing robust step durations: {e}")
            return [], []

    @staticmethod
    def compute_step_length(events, pose_data, frame_rate, side, other_side):
            """
            Compute relative step length (in pixel or unit-less values) for the given side by extracting spatial coordinates
            from the pose_data DataFrame at the event frame indices.
        
            The function:
              - Converts event times (in seconds) to frame indices using the frame rate.
              - Looks up the coordinates for the foot index marker on the specified side 
                (expected as ('left_foot_index', 'x') and ('left_foot_index', 'y') for example).
              - Computes the Euclidean distance between the markers of the two sides.
            """
            try:
                # Retrieve event times for heel strikes
                hs_side_series = events[f'HS_{side}']
                hs_other_series = events[f'HS_{other_side}']
        
                # Initialize result with NaNs.
                result = pd.Series(np.nan, index=events.index)
        
                # Process only rows with valid (finite) event times.
                valid_mask = hs_side_series.notna() & hs_other_series.notna()
                if valid_mask.sum() == 0:
                    return result
        
                valid_hs_side = hs_side_series[valid_mask]
                valid_hs_other = hs_other_series[valid_mask]
        
                # Convert event times (seconds) to frame indices.
                hs_side_indices = (valid_hs_side * frame_rate).round().astype(int)
                hs_other_indices = (valid_hs_other * frame_rate).round().astype(int)
        
                try:
                    # Try to extract coordinates assuming MultiIndex columns.
                    coords_side = pose_data.loc[hs_side_indices, (f'{side}_foot_index', ['x', 'y'])]
                    coords_other = pose_data.loc[hs_other_indices, (f'{other_side}_foot_index', ['x', 'y'])]
                except Exception as multiindex_e:
                    logger.exception(f"Error extracting coordinates using MultiIndex: {multiindex_e}")
                    # Fallback: try flat column names and rename columns.
                    coords_side = pose_data.loc[hs_side_indices, [f'{side}_foot_index_x', f'{side}_foot_index_y']]
                    coords_other = pose_data.loc[hs_other_indices, [f'{other_side}_foot_index_x', f'{other_side}_foot_index_y']]
                    coords_side.columns = ['x', 'y']
                    coords_other.columns = ['x', 'y']
        
                # Log the first coordinate pair for debugging.
                if len(hs_side_indices) > 0 and len(hs_other_indices) > 0:
                    first_left_frame = hs_side_indices.iloc[0] if hasattr(hs_side_indices, "iloc") else hs_side_indices[0]
                    first_right_frame = hs_other_indices.iloc[0] if hasattr(hs_other_indices, "iloc") else hs_other_indices[0]
                    try:
                        if isinstance(coords_side.columns, pd.MultiIndex):
                            left_x = coords_side.xs('x', axis=1, level=1).iloc[0]
                            left_y = coords_side.xs('y', axis=1, level=1).iloc[0]
                            right_x = coords_other.xs('x', axis=1, level=1).iloc[0]
                            right_y = coords_other.xs('y', axis=1, level=1).iloc[0]
                        else:
                            left_x = coords_side['x'].iloc[0]
                            left_y = coords_side['y'].iloc[0]
                            right_x = coords_other['x'].iloc[0]
                            right_y = coords_other['y'].iloc[0]
                        logger.debug("Left foot index (frame {}): x={}, y={}".format(first_left_frame, left_x, left_y))
                        logger.debug("Right foot index (frame {}): x={}, y={}".format(first_right_frame, right_x, right_y))
                    except Exception as e:
                        logger.exception("Error while logging coordinate values: %s", str(e))
        
                # If columns are a MultiIndex, extract 'x' and 'y' and squeeze to 1D.
                if isinstance(coords_side.columns, pd.MultiIndex):
                    x_side = coords_side.xs('x', axis=1, level=1).squeeze()
                    y_side = coords_side.xs('y', axis=1, level=1).squeeze()
                    x_other = coords_other.xs('x', axis=1, level=1).squeeze()
                    y_other = coords_other.xs('y', axis=1, level=1).squeeze()
                else:
                    x_side = coords_side['x']
                    y_side = coords_side['y']
                    x_other = coords_other['x']
                    y_other = coords_other['y']
        
                # Compute Euclidean distance and flatten the result.
                step_length_pixels = np.sqrt((x_side.values - x_other.values) ** 2 +
                                              (y_side.values - y_other.values) ** 2).flatten()
        
                result[valid_mask] = step_length_pixels
                return result
            except Exception as e:
                logger.exception(f"Error computing step length: {e}")
                return pd.Series(np.nan, index=events.index)
