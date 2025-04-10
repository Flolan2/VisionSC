import os
import json
import numpy as np
import pandas as pd
import logging

from modules.gait.my_utils.gait_parameters import prepare_gait_dataframe
from modules.gait.my_utils.helpers import save_csv

# Configure logger for this specific file - ensure messages are visible
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) # Set level to DEBUG for this logger

def ensure_multiindex(pose_data):
    """
    If pose_data has flat column names, convert them to a MultiIndex by splitting on the last underscore.
    For example, "left_foot_index_x" becomes ("left_foot_index", "x").
    """
    if not isinstance(pose_data.columns, pd.MultiIndex):
        logger.debug("Columns are not MultiIndex, attempting conversion.")
        new_columns = []
        for col in pose_data.columns:
            if '_' in col:
                parts = col.rsplit('_', 1)
                if len(parts) == 2: # Ensure split is successful
                    new_columns.append((parts[0], parts[1]))
                else:
                    logger.warning(f"Could not split column '{col}' into MultiIndex tuple.")
                    new_columns.append((col, '')) # Fallback
            else:
                new_columns.append((col, ''))
        pose_data.columns = pd.MultiIndex.from_tuples(new_columns, names=['marker', 'coord']) # Add names
    return pose_data


class GaitParameters:
    @staticmethod
    def compute_parameters(events, pose_data, rotated_pose_data, frame_rate, save_path=None): # Added rotated_pose_data argument
        """
        Computes gait parameters for left and right sides using rotated data for step length.
        """
        logger.debug("--- Starting GaitParameters.compute_parameters ---")
        if isinstance(events, dict):
            logger.debug("Input 'events' is a dict, converting to DataFrame.")
            events = pd.DataFrame(events)

        # Ensure pose_data has MultiIndex columns (original, non-rotated for potential other uses)
        logger.debug("Ensuring original pose_data has MultiIndex columns.")
        pose_data = ensure_multiindex(pose_data)
        # Ensure rotated_pose_data also has MultiIndex columns
        logger.debug("Ensuring rotated_pose_data has MultiIndex columns.")
        rotated_pose_data = ensure_multiindex(rotated_pose_data)


        logger.debug(f"Input 'events' DataFrame shape: {events.shape}")
        logger.debug(f"Input 'events' DataFrame head:\n{events.head().to_string()}")

        gait_df = prepare_gait_dataframe()
        logger.debug("Initialized empty gait_df.")

        # --- Stride Duration ---
        for side in ['left', 'right']:
            try:
                hs_times = events[f'HS_{side}'].dropna().values
                logger.debug(f"Calculating stride_duration for {side}. Found {len(hs_times)} HS events.")
                if len(hs_times) > 1:
                    stride_durations = np.diff(hs_times)
                    # Pad with NaN to match potential length of other parameters
                    gait_df[(side, 'stride_duration')] = pd.Series(stride_durations, index=range(len(stride_durations)))
                else:
                    logger.debug(f"Not enough HS events for {side} to calculate stride duration.")
                    gait_df[(side, 'stride_duration')] = pd.Series(dtype=float) # Ensure correct dtype for empty series
            except KeyError:
                 logger.warning(f"Column 'HS_{side}' not found in events DataFrame for stride duration.")
            except Exception as e:
                logger.exception(f"Error computing stride_duration for {side}: {e}")

        # --- Step Duration ---
        logger.debug("--- Preparing for Step Duration Calculation ---")
        logger.debug("Initial Event Counts (non-NaN) from 'events' DataFrame:")
        logger.debug(f"\n{events.notna().sum().to_string()}")
        try:
            logger.debug("Calling compute_robust_step_durations...")
            # Use lower threshold for split-belt
            left_steps, right_steps = GaitParameters.compute_robust_step_durations(events, min_valid_duration=0.1) # Lowered threshold
            logger.debug(f"compute_robust_step_durations returned: Left steps count={len(left_steps)}, Right steps count={len(right_steps)}")
            # Pad with NaN to match potential length of other parameters
            max_steps = max(len(left_steps), len(right_steps))
            gait_df[('left', 'step_duration')] = pd.Series(left_steps, index=range(len(left_steps))).reindex(range(max_steps))
            gait_df[('right', 'step_duration')] = pd.Series(right_steps, index=range(len(right_steps))).reindex(range(max_steps))
        except Exception as e:
            logger.exception(f"Error computing robust step_duration: {e}")

        # --- Cadence ---
        for side in ['left', 'right']:
            try:
                if (side, 'step_duration') in gait_df.columns:
                    step_duration_series = gait_df[(side, 'step_duration')].dropna() # Use dropna before calculation
                    logger.debug(f"Calculating cadence for {side}. Step duration series length: {len(step_duration_series)}")
                    if not step_duration_series.empty:
                         # Calculate cadence only for valid step durations
                         cadence_values = 60 / step_duration_series.replace(0, np.nan)
                         # Assign back using the original index of the valid durations
                         gait_df.loc[step_duration_series.index, (side, 'cadence')] = cadence_values
                    else:
                         gait_df[(side, 'cadence')] = pd.Series(dtype=float)
                else:
                     logger.warning(f"'{side}, step_duration' column missing for cadence calculation.")
            except Exception as e:
                logger.exception(f"Error computing cadence for {side}: {e}")


        # --- Step Length (Revised Method) ---
        logger.debug("--- Calculating Revised Step Length ---")
        try:
            logger.debug("Calling compute_step_length_revised...")
            # Pass the ROTATED pose data here
            left_sl, right_sl = GaitParameters.compute_step_length_revised(events, rotated_pose_data, frame_rate)
            # Pad with NaN to match potential length
            max_sl = max(len(left_sl), len(right_sl))
            gait_df[('left', 'step_length')] = pd.Series(left_sl, index=range(len(left_sl))).reindex(range(max_sl))
            gait_df[('right', 'step_length')] = pd.Series(right_sl, index=range(len(right_sl))).reindex(range(max_sl))
            logger.debug(f"Assigned revised step lengths: Left count={len(left_sl)}, Right count={len(right_sl)}")
        except Exception as e:
            logger.exception(f"Error computing revised step_length: {e}")


        # --- Stride Length (Uses revised step lengths now) ---
        for side in ['left', 'right']:
            try:
                 if (side, 'step_length') in gait_df.columns:
                      logger.debug(f"Calculating stride_length for {side}.")
                      step_len = gait_df[(side, 'step_length')]
                      # Need step length from the other side for stride length
                      other_side = 'left' if side == 'right' else 'right'
                      if (other_side, 'step_length') in gait_df.columns:
                           step_len_other = gait_df[(other_side, 'step_length')]
                           # Align before adding: Stride = Current Step + Next Step of *other* foot starting from current foot's HS
                           # Simpler approximation: Left Stride Length ~ Left Step Length + Subsequent Right Step Length
                           # Align step_len (current side) with step_len_other shifted appropriately
                           # This requires careful index alignment based on event sequence, complex.
                           # Common approximation: Stride Length = Step Length + contralateral Step Length (of the *same* step pair)
                           # Let's try simple addition first, assuming indices roughly align after padding
                           aligned_sl, aligned_sl_other = step_len.align(step_len_other, join='left') # Align based on current side's steps
                           gait_df[(side, 'stride_length')] = aligned_sl + aligned_sl_other # Check if this makes sense biomechanically
                           # A potentially more robust stride length comes directly from displacement between consecutive same-foot HS events
                           # Consider adding a direct stride length calculation if this approx is poor.
                      else:
                            logger.warning(f"'{other_side}, step_length' column missing for stride length calculation of {side}.")
                 else:
                      logger.warning(f"'{side}, step_length' column missing for stride length calculation.")
            except Exception as e:
                logger.exception(f"Error computing stride_length for {side}: {e}")

        # --- Gait Speed ---
        for side in ['left', 'right']:
            try:
                if (side, 'stride_length') in gait_df.columns and (side, 'stride_duration') in gait_df.columns:
                    logger.debug(f"Calculating gait_speed for {side}.")
                    # Align before division
                    sl, sd = gait_df[(side, 'stride_length')].align(gait_df[(side, 'stride_duration')], join='inner')
                    gait_df.loc[sl.index, (side, 'gait_speed')] = sl / sd.replace(0, np.nan)
                else:
                     logger.warning(f"Missing stride_length or stride_duration for {side} gait speed calculation.")
            except Exception as e:
                logger.exception(f"Error computing gait_speed for {side}: {e}")

        # --- Swing Time ---
        for side in ['left', 'right']:
            try:
                if f'HS_{side}' in events.columns and f'TO_{side}' in events.columns:
                     logger.debug(f"Calculating swing time for {side}.")
                     hs_times = events[f'HS_{side}'].dropna()
                     to_times = events[f'TO_{side}'].dropna()

                     swing_times = []
                     # Find the TO immediately preceding each HS (except the first HS)
                     for i in range(len(hs_times)):
                         current_hs_time = hs_times.iloc[i]
                         # Find TOs that occurred *before* this HS
                         preceding_tos = to_times[to_times < current_hs_time]
                         if not preceding_tos.empty:
                              # Find the *last* TO before this HS
                              last_to_time = preceding_tos.iloc[-1]
                              # Swing time = HS_time - Previous_TO_time (of same foot)
                              # This isn't right. Swing = Time from TO to *next* HS of *same* foot.

                              # Let's retry: Align TO with next HS
                              current_to_time = to_times.iloc[i] if i < len(to_times) else np.nan
                              if pd.notna(current_to_time):
                                   # Find HS events *after* this TO
                                   following_hs = hs_times[hs_times > current_to_time]
                                   if not following_hs.empty:
                                        next_hs_time = following_hs.iloc[0]
                                        swing = next_hs_time - current_to_time
                                        swing_times.append(swing)
                                   else:
                                        swing_times.append(np.nan) # No next HS found
                              else:
                                   swing_times.append(np.nan) # No current TO

                         else:
                              swing_times.append(np.nan) # No preceding TO found


                     # Pad with NaNs
                     max_len = len(gait_df) # Use length of DataFrame index
                     gait_df[(side, 'swing')] = pd.Series(swing_times, index=range(len(swing_times))).reindex(range(max_len))

                else:
                     logger.warning(f"Missing HS_{side} or TO_{side} for {side} swing time calculation.")
            except Exception as e:
                logger.exception(f"Error computing swing for {side}: {e}")


        # --- Initial Double Support ---
        # IDS = Time from current HS to opposite foot TO
        for side in ['left', 'right']:
            other_side = 'left' if side == 'right' else 'right'
            try:
                if f'HS_{side}' in events.columns and f'TO_{other_side}' in events.columns:
                    logger.debug(f"Calculating initial_double_support for {side}.")
                    hs_times = events[f'HS_{side}'].dropna()
                    to_other_times = events[f'TO_{other_side}'].dropna()
                    ids_times = []

                    for i in range(len(hs_times)):
                         current_hs_time = hs_times.iloc[i]
                         # Find the first TO_other that occurs *after* current_hs_time
                         following_to_other = to_other_times[to_other_times >= current_hs_time] # Use >= to include simultaneous
                         if not following_to_other.empty:
                              first_following_to_other_time = following_to_other.iloc[0]
                              ids = first_following_to_other_time - current_hs_time
                              ids_times.append(ids)
                         else:
                              ids_times.append(np.nan) # No opposite TO found after HS

                    # Pad with NaNs
                    max_len = len(gait_df)
                    gait_df[(side, 'initial_double_support')] = pd.Series(ids_times, index=range(len(ids_times))).reindex(range(max_len))

                else:
                     logger.warning(f"Missing HS_{side} or TO_{other_side} for {side} initial double support calculation.")
            except Exception as e:
                logger.exception(f"Error computing initial_double_support for {side}: {e}")


        # --- Asymmetry ---
        # Make sure to recalculate after all base parameters are computed and assigned
        logger.debug("--- Calculating Asymmetry ---")
        for feature in ['stride_duration', 'step_duration', 'cadence', 'swing', 'initial_double_support', 'step_length', 'stride_length', 'gait_speed']:
            left_col = ('left', feature)
            right_col = ('right', feature)
            asym_col = ('asymmetry', feature)
            try:
                if left_col in gait_df.columns and right_col in gait_df.columns:
                    logger.debug(f"Calculating asymmetry for {feature}.")
                    left_data = gait_df[left_col]
                    right_data = gait_df[right_col]
                    # Align data before calculation using the DataFrame index
                    aligned_left, aligned_right = left_data.align(right_data, join='outer')

                    # Calculate asymmetry only where both sides have data
                    valid_mask = aligned_left.notna() & aligned_right.notna()
                    if valid_mask.any():
                         diff = abs(aligned_left[valid_mask] - aligned_right[valid_mask])
                         # Denominator: max(|L|, |R|) or (L+R)/2 ? Using max(|L|,|R|)
                         max_val = np.maximum(np.abs(aligned_left[valid_mask]), np.abs(aligned_right[valid_mask]))
                         # Avoid division by zero
                         asymmetry_percent = (diff / max_val.replace(0, np.nan) * 100)
                         gait_df.loc[valid_mask, asym_col] = asymmetry_percent # Assign back using mask
                    else:
                         logger.debug(f"No overlapping valid data for {feature} to calculate asymmetry.")

                else:
                     # Check if feature exists under 'asymmetry' level from prepare_gait_dataframe
                     if asym_col in gait_df.columns:
                           logger.warning(f"Missing left or right data for feature '{feature}', cannot calculate asymmetry. Column {asym_col} exists but will be NaN.")
                     else:
                           logger.warning(f"Missing left or right data for feature '{feature}', cannot calculate asymmetry. Column {asym_col} does not exist.")

            except Exception as e:
                logger.exception(f"Error computing asymmetry for feature {feature}: {e}")


        logger.debug(f"Final gait_df head before return:\n{gait_df.head().to_string()}")
        logger.debug(f"Final gait_df describe before return:\n{gait_df.describe().to_string()}")
        logger.debug("--- Finished GaitParameters.compute_parameters ---")
        return gait_df


    @staticmethod
    def compute_robust_step_durations(events, min_valid_duration=0.1): # Lowered threshold
        """
        Computes robust step durations by merging left and right heel strikes
        and pairing them based on the actual sequence of events.
        Includes DEBUG logging.
        """
        logger.debug("--- Starting compute_robust_step_durations ---")
        left_step_durations = []
        right_step_durations = []
        try:
            if 'HS_left' not in events.columns or 'HS_right' not in events.columns:
                 logger.error("HS_left or HS_right columns missing in events DataFrame.")
                 return [], []

            left_times = events['HS_left'].dropna().values
            right_times = events['HS_right'].dropna().values
            logger.debug(f"Initial counts before combining: Left HS={len(left_times)}, Right HS={len(right_times)}")

            if len(left_times) == 0 and len(right_times) == 0:
                 logger.warning("No valid HS events found for either side.")
                 return [], []

            left_df = pd.DataFrame({'time': left_times, 'side': 'left'})
            right_df = pd.DataFrame({'time': right_times, 'side': 'right'})
            combined = pd.concat([left_df, right_df]).sort_values('time').reset_index(drop=True)
            logger.debug(f"Combined and sorted HS event count: {len(combined)}")
            # logger.debug(f"Combined DataFrame head:\n{combined.head().to_string()}") # Can be verbose

            for i in range(len(combined) - 1):
                current = combined.iloc[i]
                nxt = combined.iloc[i + 1]
                duration = nxt['time'] - current['time']
                logger.debug(f"  Index {i}: Side={current['side']}, NextSide={nxt['side']}, Duration={duration:.4f}")

                if duration < min_valid_duration:
                    logger.debug(f"    -> Duration below threshold {min_valid_duration}, skipping.")
                    continue

                if current['side'] == 'left' and nxt['side'] == 'right':
                    left_step_durations.append(duration)
                    # logger.debug("    -> Added to LEFT step durations")
                elif current['side'] == 'right' and nxt['side'] == 'left':
                    right_step_durations.append(duration)
                    # logger.debug("    -> Added to RIGHT step durations")
                else:
                    logger.debug(f"    -> Skipping duration between two consecutive {current['side']} heel strikes.")

            logger.debug(f"Final computed step duration counts: Left={len(left_step_durations)}, Right={len(right_step_durations)}")
            return left_step_durations, right_step_durations

        except Exception as e:
            logger.exception(f"Error computing robust step durations: {e}")
            return [], []


    @staticmethod
    def compute_step_length_revised(events, rotated_pose_data, frame_rate):
        """
        Computes step length based on forward displacement (Z-axis) of the ankle during a step
        in the ROTATED coordinate system.
        Returns two lists: left_step_lengths, right_step_lengths
        """
        logger.debug("--- Starting compute_step_length_revised ---")
        left_step_lengths = []
        right_step_lengths = []
        try:
            # Ensure required columns exist in rotated_pose_data
            ankle_l_z = ('left_ankle', 'z')
            ankle_r_z = ('right_ankle', 'z')
            if not (ankle_l_z in rotated_pose_data.columns and ankle_r_z in rotated_pose_data.columns):
                 missing = [col for col in [ankle_l_z, ankle_r_z] if col not in rotated_pose_data.columns]
                 logger.error(f"Missing required ankle Z columns in rotated_pose_data for revised step length: {missing}")
                 return [], []

            # Get all heel strike times and sort them
            if 'HS_left' not in events.columns or 'HS_right' not in events.columns:
                 logger.error("HS_left or HS_right columns missing in events DataFrame.")
                 return [], []
            left_times = events['HS_left'].dropna()
            right_times = events['HS_right'].dropna()

            if left_times.empty and right_times.empty:
                 logger.warning("No HS events found for step length calculation.")
                 return [], []

            left_df = pd.DataFrame({'time': left_times, 'side': 'left'})
            right_df = pd.DataFrame({'time': right_times, 'side': 'right'})
            combined_hs = pd.concat([left_df, right_df]).sort_values('time').reset_index(drop=True)
            logger.debug(f"Combined HS events for step length: {len(combined_hs)}")

            max_frame_index = len(rotated_pose_data) - 1

            # Iterate through consecutive heel strikes
            for i in range(len(combined_hs) - 1):
                prev_hs = combined_hs.iloc[i]
                curr_hs = combined_hs.iloc[i + 1]

                # Ensure consecutive HS are from opposite feet
                if prev_hs['side'] == curr_hs['side']:
                    logger.debug(f"Skipping step {i}: Consecutive HS from same side ({prev_hs['side']}).")
                    continue

                # Determine which foot took the step (the one landing at curr_hs)
                stepping_side = curr_hs['side']
                ankle_marker_col = (f'{stepping_side}_ankle', 'z')

                # Get frame indices for the start (prev_hs) and end (curr_hs) of the step
                frame_start = int(round(prev_hs['time'] * frame_rate))
                frame_end = int(round(curr_hs['time'] * frame_rate))

                # Ensure frames are valid and end is after start
                if not (0 <= frame_start <= max_frame_index and 0 <= frame_end <= max_frame_index):
                    logger.warning(f"Invalid frame indices [{frame_start}, {frame_end}] (max: {max_frame_index}) for step {i}. Skipping.")
                    continue
                if frame_start >= frame_end:
                    logger.warning(f"Step end frame ({frame_end}) not after start frame ({frame_start}) for step {i}. Skipping.")
                    continue

                # Get the Z-coordinate of the stepping foot's ANKLE at the start and end frames
                try:
                    # Use .iloc for integer position indexing if loc fails with calculated frames
                    z_start = rotated_pose_data.iloc[frame_start][ankle_marker_col]
                    z_end = rotated_pose_data.iloc[frame_end][ankle_marker_col]
                except (KeyError, IndexError) as ke:
                     logger.warning(f"Could not get Z coordinate for {ankle_marker_col} at frame {frame_start} or {frame_end}. Error: {ke}. Skipping step length.")
                     continue

                 # Check for NaN values
                if pd.isna(z_start) or pd.isna(z_end):
                    logger.warning(f"NaN value encountered for Z coordinate at frame {frame_start} or {frame_end}. Skipping step length.")
                    continue

                # Calculate step length as the difference in Z coordinate (forward displacement)
                step_length = z_end - z_start
                logger.debug(f"  Step {i}: SteppingSide={stepping_side}, Frames=[{frame_start}-{frame_end}], Z_start={z_start:.3f}, Z_end={z_end:.3f}, StepLength={step_length:.3f}")

                # Append to the correct list
                if stepping_side == 'left':
                    left_step_lengths.append(step_length)
                elif stepping_side == 'right':
                    right_step_lengths.append(step_length)

            logger.debug(f"Finished revised step length calculation. Left count={len(left_step_lengths)}, Right count={len(right_step_lengths)}")
            return left_step_lengths, right_step_lengths

        except Exception as e:
            logger.exception(f"Error computing revised step lengths: {e}")
            return [], []

