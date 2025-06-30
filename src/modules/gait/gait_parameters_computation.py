# --- START OF FILE gait_parameters_computation.py ---

import os
import json
import numpy as np
import pandas as pd
import logging

# MODIFIED IMPORT: Add 'parameters' here
from my_utils.gait_parameters import prepare_gait_dataframe, parameters # parameters is used below

from my_utils.helpers import save_csv

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
        Adds a 'Step' column at the beginning, numbering the rows starting from 1.
        Now includes calculations for Stance, Terminal Double Support, and Total Double Support.
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

        # --- Ensure all columns from prepare_gait_dataframe exist initially ---
        for side in ['left', 'right']:
            for param_name in parameters: # Use imported parameters set
                 col_tuple = (side, param_name)
                 if col_tuple not in gait_df.columns:
                       gait_df[col_tuple] = pd.Series(dtype=float)
        for param_name in parameters:
             if '_asymmetry' in param_name:
                  base_feature = param_name.replace('_asymmetry', '')
                  col_tuple = ('asymmetry', base_feature)
                  if col_tuple not in gait_df.columns:
                     gait_df[col_tuple] = pd.Series(dtype=float)
        # -------------------------------------------------------------------


        # --- Stride Duration ---
        for side in ['left', 'right']:
            try:
                hs_times = events[f'HS_{side}'].dropna().values
                logger.debug(f"Calculating stride_duration for {side}. Found {len(hs_times)} HS events.")
                if len(hs_times) > 1:
                    stride_durations = np.diff(hs_times)
                    gait_df[(side, 'stride_duration')] = pd.Series(stride_durations, index=range(len(stride_durations)))
                else:
                    logger.debug(f"Not enough HS events for {side} to calculate stride duration.")
                    gait_df[(side, 'stride_duration')] = pd.Series(dtype=float)
            except KeyError:
                 logger.warning(f"Column 'HS_{side}' not found in events DataFrame for stride duration.")
                 gait_df[(side, 'stride_duration')] = pd.Series(dtype=float) # Ensure column exists
            except Exception as e:
                logger.exception(f"Error computing stride_duration for {side}: {e}")
                gait_df[(side, 'stride_duration')] = pd.Series(dtype=float) # Ensure column exists


        # --- Step Duration ---
        logger.debug("--- Preparing for Step Duration Calculation ---")
        logger.debug("Initial Event Counts (non-NaN) from 'events' DataFrame:")
        logger.debug(f"\n{events.notna().sum().to_string()}")
        try:
            logger.debug("Calling compute_robust_step_durations...")
            left_steps, right_steps = GaitParameters.compute_robust_step_durations(events, min_valid_duration=0.1)
            logger.debug(f"compute_robust_step_durations returned: Left steps count={len(left_steps)}, Right steps count={len(right_steps)}")
            max_steps = max(len(left_steps), len(right_steps))
            index_range = range(max_steps) if max_steps > 0 else pd.RangeIndex(0)
            gait_df[('left', 'step_duration')] = pd.Series(left_steps, index=range(len(left_steps))).reindex(index_range)
            gait_df[('right', 'step_duration')] = pd.Series(right_steps, index=range(len(right_steps))).reindex(index_range)
        except Exception as e:
            logger.exception(f"Error computing robust step_duration: {e}")
            gait_df[('left', 'step_duration')] = pd.Series(dtype=float)
            gait_df[('right', 'step_duration')] = pd.Series(dtype=float)


        # --- Cadence ---
        for side in ['left', 'right']:
            try:
                if (side, 'step_duration') in gait_df.columns:
                    step_duration_series = gait_df[(side, 'step_duration')].dropna()
                    logger.debug(f"Calculating cadence for {side}. Step duration series length: {len(step_duration_series)}")
                    if not step_duration_series.empty:
                         cadence_values = 60 / step_duration_series.replace(0, np.nan)
                         gait_df.loc[step_duration_series.index, (side, 'cadence')] = cadence_values
                    # else: column already exists from initialization
                else:
                     logger.warning(f"'{side}, step_duration' column missing for cadence calculation.")
            except Exception as e:
                logger.exception(f"Error computing cadence for {side}: {e}")


        # --- Step Length (Revised Method) ---
        logger.debug("--- Calculating Revised Step Length ---")
        try:
            logger.debug("Calling compute_step_length_revised...")
            left_sl, right_sl = GaitParameters.compute_step_length_revised(events, rotated_pose_data, frame_rate)
            max_sl = max(len(left_sl), len(right_sl))
            index_range_sl = range(max_sl) if max_sl > 0 else pd.RangeIndex(0)
            gait_df[('left', 'step_length')] = pd.Series(left_sl, index=range(len(left_sl))).reindex(index_range_sl)
            gait_df[('right', 'step_length')] = pd.Series(right_sl, index=range(len(right_sl))).reindex(index_range_sl)
            logger.debug(f"Assigned revised step lengths: Left count={len(left_sl)}, Right count={len(right_sl)}")
        except Exception as e:
            logger.exception(f"Error computing revised step_length: {e}")
            gait_df[('left', 'step_length')] = pd.Series(dtype=float)
            gait_df[('right', 'step_length')] = pd.Series(dtype=float)


        # --- Stride Length (Uses revised step lengths now) ---
        for side in ['left', 'right']:
            try:
                 if (side, 'step_length') in gait_df.columns:
                      step_len = gait_df[(side, 'step_length')]
                      other_side = 'left' if side == 'right' else 'right'
                      if (other_side, 'step_length') in gait_df.columns:
                           step_len_other = gait_df[(other_side, 'step_length')]
                           aligned_sl, aligned_sl_other = step_len.align(step_len_other, join='left') # Align based on current side's steps
                           stride_length_series = aligned_sl + aligned_sl_other
                           gait_df[(side, 'stride_length')] = stride_length_series
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
                    sl, sd = gait_df[(side, 'stride_length')].align(gait_df[(side, 'stride_duration')], join='inner')
                    speed_values = sl / sd.replace(0, np.nan)
                    gait_df.loc[speed_values.index, (side, 'gait_speed')] = speed_values
                else:
                     logger.warning(f"Missing stride_length or stride_duration for {side} gait speed calculation.")
            except Exception as e:
                logger.exception(f"Error computing gait_speed for {side}: {e}")


        # --- Swing Time ---
        # Swing = Time from TO to *next* HS of *same* foot.
        for side in ['left', 'right']:
            swing_times = []
            try:
                if f'HS_{side}' in events.columns and f'TO_{side}' in events.columns:
                     hs_times = events[f'HS_{side}'].dropna()
                     to_times = events[f'TO_{side}'].dropna()
                     if not hs_times.empty and not to_times.empty:
                         logger.debug(f"Calculating swing time for {side}.")
                         # Match each TO with the immediately following HS of the same side
                         to_indices = to_times.index
                         hs_indices = hs_times.index
                         for i in range(len(to_times)):
                             current_to_time = to_times.iloc[i]
                             # Find HS events *after* this TO
                             following_hs = hs_times[hs_times > current_to_time]
                             if not following_hs.empty:
                                  next_hs_time = following_hs.iloc[0]
                                  swing = next_hs_time - current_to_time
                                  swing_times.append(swing)
                             else:
                                  swing_times.append(np.nan) # No next HS found for this TO
                         # Assign swing times, aligning potentially with stride index
                         gait_df[(side, 'swing')] = pd.Series(swing_times, index=range(len(swing_times))) # Simple assignment for now
                     else:
                          logger.warning(f"Empty HS_{side} or TO_{side} times for {side} swing calculation.")
                else:
                     logger.warning(f"Missing HS_{side} or TO_{side} columns for {side} swing calculation.")
            except Exception as e:
                logger.exception(f"Error computing swing for {side}: {e}")


        # --- Stance Time --- NEW CALCULATION
        # Stance = Time from HS to *next* TO of *same* foot.
        for side in ['left', 'right']:
            stance_times = []
            try:
                if f'HS_{side}' in events.columns and f'TO_{side}' in events.columns:
                     hs_times = events[f'HS_{side}'].dropna()
                     to_times = events[f'TO_{side}'].dropna()
                     if not hs_times.empty and not to_times.empty:
                         logger.debug(f"Calculating stance time for {side}.")
                         # Match each HS with the immediately following TO of the same side
                         for i in range(len(hs_times)):
                             current_hs_time = hs_times.iloc[i]
                             # Find TO events *after* this HS
                             following_to = to_times[to_times > current_hs_time]
                             if not following_to.empty:
                                  next_to_time = following_to.iloc[0]
                                  stance = next_to_time - current_hs_time
                                  stance_times.append(stance)
                             else:
                                  stance_times.append(np.nan) # No next TO found for this HS
                         # Assign stance times, aligning potentially with stride index
                         gait_df[(side, 'stance')] = pd.Series(stance_times, index=range(len(stance_times))) # Simple assignment
                     else:
                          logger.warning(f"Empty HS_{side} or TO_{side} times for {side} stance calculation.")
                else:
                     logger.warning(f"Missing HS_{side} or TO_{side} columns for {side} stance calculation.")
            except Exception as e:
                logger.exception(f"Error computing stance for {side}: {e}")

        # --- Initial Double Support ---
        # IDS = Time from current HS to *next* opposite foot TO
        for side in ['left', 'right']:
            other_side = 'left' if side == 'right' else 'right'
            ids_times = []
            try:
                if f'HS_{side}' in events.columns and f'TO_{other_side}' in events.columns:
                    hs_times = events[f'HS_{side}'].dropna()
                    to_other_times = events[f'TO_{other_side}'].dropna()
                    if not hs_times.empty and not to_other_times.empty:
                        logger.debug(f"Calculating initial_double_support for {side}.")
                        # Match each HS with the immediately following opposite TO
                        for i in range(len(hs_times)):
                             current_hs_time = hs_times.iloc[i]
                             following_to_other = to_other_times[to_other_times >= current_hs_time] # Use >=
                             if not following_to_other.empty:
                                  first_following_to_other_time = following_to_other.iloc[0]
                                  ids = first_following_to_other_time - current_hs_time
                                  ids_times.append(ids)
                             else:
                                  ids_times.append(np.nan)
                        gait_df[(side, 'initial_double_support')] = pd.Series(ids_times, index=range(len(ids_times)))
                    else:
                         logger.warning(f"Empty HS_{side} or TO_{other_side} times for {side} IDS calculation.")
                else:
                     logger.warning(f"Missing HS_{side} or TO_{other_side} columns for {side} initial double support calculation.")
            except Exception as e:
                logger.exception(f"Error computing initial_double_support for {side}: {e}")

        # --- Terminal Double Support --- NEW CALCULATION
        # TDS = Time from opposite foot HS to *current* foot TO
        for side in ['left', 'right']:
            other_side = 'left' if side == 'right' else 'right'
            tds_times = []
            try:
                if f'HS_{other_side}' in events.columns and f'TO_{side}' in events.columns:
                    hs_other_times = events[f'HS_{other_side}'].dropna()
                    to_times = events[f'TO_{side}'].dropna()
                    if not hs_other_times.empty and not to_times.empty:
                        logger.debug(f"Calculating terminal_double_support for {side}.")
                        # Match each TO with the immediately preceding opposite HS
                        for i in range(len(to_times)):
                            current_to_time = to_times.iloc[i]
                            preceding_hs_other = hs_other_times[hs_other_times < current_to_time] # Strictly preceding
                            if not preceding_hs_other.empty:
                                last_preceding_hs_other_time = preceding_hs_other.iloc[-1]
                                tds = current_to_time - last_preceding_hs_other_time
                                tds_times.append(tds)
                            else:
                                tds_times.append(np.nan)
                        # Align TDS with the TO events, needs careful consideration for final structure
                        gait_df[(side, 'terminal_double_support')] = pd.Series(tds_times, index=range(len(tds_times)))
                    else:
                         logger.warning(f"Empty HS_{other_side} or TO_{side} times for {side} TDS calculation.")
                else:
                    logger.warning(f"Missing HS_{other_side} or TO_{side} columns for {side} terminal double support calculation.")
            except Exception as e:
                logger.exception(f"Error computing terminal_double_support for {side}: {e}")

        # --- Total Double Support --- NEW CALCULATION
        # Total DS = Initial Double Support + Terminal Double Support
        for side in ['left', 'right']:
             try:
                 if (side, 'initial_double_support') in gait_df.columns and (side, 'terminal_double_support') in gait_df.columns:
                      logger.debug(f"Calculating total double_support for {side}.")
                      # Align IDS and TDS. Assuming they correspond to the same cycle/stride number.
                      ids_s, tds_s = gait_df[(side, 'initial_double_support')].align(gait_df[(side, 'terminal_double_support')], join='outer')
                      total_ds = ids_s + tds_s
                      gait_df[(side, 'double_support')] = total_ds
                 else:
                      logger.warning(f"Missing IDS or TDS for {side} total double support calculation.")
             except Exception as e:
                  logger.exception(f"Error computing total double_support for {side}: {e}")


        # --- Asymmetry ---
        logger.debug("--- Calculating Asymmetry ---")
        # List of base features for which asymmetry is calculated
        asymmetry_base_features = [p.replace('_asymmetry', '') for p in parameters if '_asymmetry' in p]

        for feature in asymmetry_base_features:
            # Handle special case 'double_support' vs 'initial_double_support'/'terminal_double_support' if needed
            # Currently assuming asymmetry is calculated for all base params listed here + stance/TDS/DS
            if feature not in ['stride_duration', 'step_duration', 'cadence', 'swing', 'stance', 'initial_double_support', 'terminal_double_support', 'double_support', 'step_length', 'stride_length', 'gait_speed']:
                continue # Skip if not a feature we calculate L/R for

            left_col = ('left', feature)
            right_col = ('right', feature)
            asym_col = ('asymmetry', feature) # Direct mapping based on name
            try:
                if left_col in gait_df.columns and right_col in gait_df.columns:
                    left_data = gait_df[left_col]
                    right_data = gait_df[right_col]
                    if left_data.notna().any() and right_data.notna().any(): # Check if there's any data
                        logger.debug(f"Calculating asymmetry for {feature}.")
                        aligned_left, aligned_right = left_data.align(right_data, join='outer')
                        valid_mask = aligned_left.notna() & aligned_right.notna()
                        if valid_mask.any():
                             diff = abs(aligned_left[valid_mask] - aligned_right[valid_mask])
                             max_val = np.maximum(np.abs(aligned_left[valid_mask]), np.abs(aligned_right[valid_mask]))
                             asymmetry_percent = (diff / max_val.replace(0, np.nan) * 100).fillna(0)
                             gait_df.loc[valid_mask, asym_col] = asymmetry_percent
                        else:
                             logger.debug(f"No overlapping valid data for {feature} to calculate asymmetry.")
                    else:
                         logger.debug(f"Not enough non-NaN data for {feature} asymmetry calculation.")
                else:
                     logger.warning(f"Missing left or right data for feature '{feature}', cannot calculate asymmetry.")

            except Exception as e:
                logger.exception(f"Error computing asymmetry for feature {feature}: {e}")


        # --- Reindex DataFrame to ensure consistent length across columns ---
        max_len = 0
        for col in gait_df.columns:
            max_len = max(max_len, gait_df[col].last_valid_index() + 1 if gait_df[col].last_valid_index() is not None else 0)
        final_index = range(max_len) if max_len > 0 else pd.RangeIndex(0)
        gait_df = gait_df.reindex(final_index)


        # --- Clean up empty rows potentially created by reindexing ---
        gait_df.dropna(axis=0, how='all', inplace=True)
        gait_df.reset_index(drop=True, inplace=True) # Reset index after dropping rows


        # ************************************************
        # --- ADD 'Step' COLUMN at the beginning ---
        # ************************************************
        if not gait_df.empty:
            step_numbers = gait_df.index + 1
            gait_df.insert(0, 'Step', step_numbers)
            logger.debug("Added 'Step' column to gait_df.")
            gait_df['Step'] = gait_df['Step'].astype(int)
        else:
            gait_df['Step'] = pd.Series(dtype=int)
            if 'Step' in gait_df.columns:
                 cols = ['Step'] + [col for col in gait_df.columns if col != 'Step']
                 gait_df = gait_df[cols]
            logger.debug("gait_df was empty, added an empty 'Step' column.")


        logger.debug(f"Final gait_df head before return:\n{gait_df.head().to_string()}")
        logger.debug(f"Final gait_df describe before return:\n{gait_df.describe().to_string()}")
        logger.debug("--- Finished GaitParameters.compute_parameters ---")
        return gait_df


    @staticmethod
    def compute_robust_step_durations(events, min_valid_duration=0.1):
        # ... (no changes needed in this function) ...
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
            # Handle empty DataFrames before concat
            if left_df.empty and right_df.empty:
                 combined = pd.DataFrame(columns=['time', 'side'])
            elif left_df.empty:
                 combined = right_df.sort_values('time').reset_index(drop=True)
            elif right_df.empty:
                 combined = left_df.sort_values('time').reset_index(drop=True)
            else:
                 combined = pd.concat([left_df, right_df]).sort_values('time').reset_index(drop=True)

            logger.debug(f"Combined and sorted HS event count: {len(combined)}")

            for i in range(len(combined) - 1):
                current = combined.iloc[i]
                nxt = combined.iloc[i + 1]
                duration = nxt['time'] - current['time']
                logger.debug(f"  Index {i}: Side={current['side']}, NextSide={nxt['side']}, Time={current['time']:.3f}, NextTime={nxt['time']:.3f}, Duration={duration:.4f}")

                if duration < min_valid_duration:
                    logger.debug(f"    -> Duration below threshold {min_valid_duration}, skipping.")
                    continue

                # A 'left step' is HS_right to HS_left
                if current['side'] == 'right' and nxt['side'] == 'left':
                    left_step_durations.append(duration)
                    logger.debug(f"    -> Added to LEFT step durations (RHS @ {current['time']:.3f} -> LHS @ {nxt['time']:.3f})")
                # A 'right step' is HS_left to HS_right
                elif current['side'] == 'left' and nxt['side'] == 'right':
                    right_step_durations.append(duration)
                    logger.debug(f"    -> Added to RIGHT step durations (LHS @ {current['time']:.3f} -> RHS @ {nxt['time']:.3f})")
                else:
                    logger.debug(f"    -> Skipping duration between two consecutive {current['side']} heel strikes.")

            logger.debug(f"Final computed step duration counts: Left={len(left_step_durations)}, Right={len(right_step_durations)}")
            return left_step_durations, right_step_durations

        except Exception as e:
            logger.exception(f"Error computing robust step durations: {e}")
            return [], []


    @staticmethod
    def compute_step_length_revised(events, rotated_pose_data, frame_rate):
        # ... (no changes needed in this function) ...
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

            # Handle empty DataFrames before concat
            if left_df.empty and right_df.empty:
                 combined_hs = pd.DataFrame(columns=['time', 'side'])
            elif left_df.empty:
                 combined_hs = right_df.sort_values('time').reset_index(drop=True)
            elif right_df.empty:
                 combined_hs = left_df.sort_values('time').reset_index(drop=True)
            else:
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
                # Determine the *other* foot (stance foot at start of step)
                stance_side = prev_hs['side']

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

                # Step Length Definition: Forward distance between ankles at moment of heel strike
                ankle_marker_left = ('left_ankle', 'z')
                ankle_marker_right = ('right_ankle', 'z')

                try:
                    # Get Z-coord of BOTH ankles at the end of the step (moment of curr_hs)
                    z_left_ankle_at_end = rotated_pose_data.iloc[frame_end][ankle_marker_left]
                    z_right_ankle_at_end = rotated_pose_data.iloc[frame_end][ankle_marker_right]

                except (KeyError, IndexError) as ke:
                     logger.warning(f"Could not get Z coordinate for ankles at frame {frame_end}. Error: {ke}. Skipping step length for step {i}.")
                     continue

                 # Check for NaN values
                if pd.isna(z_left_ankle_at_end) or pd.isna(z_right_ankle_at_end):
                    logger.warning(f"NaN value encountered for ankle Z coordinate at frame {frame_end}. Skipping step length for step {i}.")
                    continue

                # Calculate step length as the absolute difference in Z coordinate between ankles at HS
                step_length = abs(z_left_ankle_at_end - z_right_ankle_at_end) # Absolute difference

                logger.debug(f"  Step {i}: SteppingSide={stepping_side}, StanceSide={stance_side}, FrameEnd={frame_end}, ZL={z_left_ankle_at_end:.3f}, ZR={z_right_ankle_at_end:.3f}, StepLength={step_length:.3f}")

                # Append to the correct list based on the *stepping* foot
                if stepping_side == 'left':
                    left_step_lengths.append(step_length)
                elif stepping_side == 'right':
                    right_step_lengths.append(step_length)

            logger.debug(f"Finished revised step length calculation. Left count={len(left_step_lengths)}, Right count={len(right_step_lengths)}")
            return left_step_lengths, right_step_lengths

        except Exception as e:
            logger.exception(f"Error computing revised step lengths: {e}")
            return [], []