#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 22 21:40:20 2025

@author: Lange_L
"""

import os
import json
import logging
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Import YOLO cropper for video processing
from modules.yolo_cropper import YOLOCropper

# Helpers for pose data loading and frame rate retrieval
from modules.gait.my_utils.helpers import load_csv, get_frame_rate, save_csv

# Import visualization helpers
from modules.gait.my_utils.plotting import (
    butter_lowpass_filter,
    detect_extremas,
    plot_combined_extremas_and_toe, # Keep this for saving the plot non-interactively
    plot_parameter_timeseries,
    plot_parameter_distributions,
    plot_left_right_comparison,
    plot_fog_visualization
)
# REMOVE import of prompt_visualisation
# from modules.gait.my_utils.prompt_visualisation import prompt_visualisation

# Import additional modules used by the gait pipeline
from modules.pose_estimation import PoseEstimator
from modules.gait.gait_preprocessing import Preprocessor
from modules.gait.gait_event_detection import EventDetector # Needs modification to return rotated_pose_data
from modules.gait.gait_parameters_computation import GaitParameters
from modules.gait.gait_freezing_detector import FreezingDetector

logger = logging.getLogger(__name__)

# GaitPipeline Class definition remains the same as the last version you posted...
# Add storage for rotated_pose_data if not already present
class GaitPipeline:
    def __init__(self, input_path, config):
        self.input_path = input_path
        self.config = config
        self.pose_data = None
        self.rotated_pose_data = None # ADDED: To store rotated data
        self.frame_rate = None
        self.events = None
        self.gait_params = None
        self.fog_events = None
        self.forward_displacement = None
        self.logger = logging.getLogger(f"{__name__}.GaitPipeline")

    def load_input(self):
        # ... (no changes needed here) ...
        """Loads input data (CSV or Video) and determines frame rate."""
        self.logger.info(f"Loading input from: {self.input_path}")
        if self.input_path.lower().endswith(".csv"):
            self.pose_data = load_csv(file_path=self.input_path)
            if self.pose_data is None:
                 raise ValueError(f"Failed to load CSV: {self.input_path}")
            self.frame_rate = get_frame_rate(file_path=self.input_path) # Assumes metadata exists for CSV
            if self.frame_rate is None:
                 self.logger.warning(f"Could not determine frame rate from metadata for CSV: {self.input_path}. Using default from config.")
                 self.frame_rate = self.config.get('event_detection', {}).get('frame_rate', 30) # Fallback
            self.logger.info(f"Loaded CSV data. Frame rate: {self.frame_rate} Hz")

        elif self.input_path.lower().endswith((".mp4", ".mov")):
            pose_estimator = PoseEstimator(config=self.config)
            self.pose_data, self.frame_rate = pose_estimator.process_video(video_path=self.input_path)
            if self.pose_data is None:
                raise ValueError(f"Pose estimation failed for video: {self.input_path}")
            if self.frame_rate is None or self.frame_rate <= 0:
                 self.logger.warning(f"Invalid frame rate {self.frame_rate} from pose estimator for {self.input_path}. Using default from config.")
                 self.frame_rate = self.config.get('event_detection', {}).get('frame_rate', 30) # Fallback
            # Ensure numeric conversion after pose estimation
            self.pose_data = self.pose_data.apply(pd.to_numeric, errors="coerce")
            self.logger.info(f"Processed video. Frame rate: {self.frame_rate} Hz. Pose data shape: {self.pose_data.shape}")

        else:
            raise ValueError(f"Unsupported input format: {self.input_path}. Use .mp4/.mov or .csv.")

        return self.pose_data, self.frame_rate


    def preprocess(self):
        # ... (no changes needed here) ...
        """Preprocesses the loaded pose data."""
        if self.pose_data is None:
            raise ValueError("Cannot preprocess, pose_data is not loaded.")
        self.logger.info("Starting preprocessing...")
        preprocessor = Preprocessor(pose_data=self.pose_data)
        window_size = self.config.get("preprocessing", {}).get("median_filter_window", 11) # Default window size
        self.pose_data = preprocessor.preprocess(window_size=window_size)
        self.logger.info("Preprocessing complete.")
        return self.pose_data


    def detect_events(self):
        # MODIFIED: Store rotated_pose_data
        if self.pose_data is None:
            raise ValueError("Cannot detect events, pose_data is not loaded.")
        if self.frame_rate is None:
            raise ValueError("Cannot detect events, frame_rate is not set.")

        self.logger.info("Starting event detection...")
        # Assuming EventDetector has been modified to return (events, rotated_data)
        # OR modify EventDetector's detect_heel_toe_events to store rotated_data internally
        # Let's modify EventDetector's method to return both
        detector = EventDetector(
            **self.config.get("event_detection", {}),
            input_path=self.input_path,
            frame_rate=self.frame_rate,
            config=self.config
            )
        # *** This line assumes EventDetector.detect_heel_toe_events is changed ***
        # *** to return a tuple: (events_df, rotated_pose_df) ***
        self.events, self.rotated_pose_data = detector.detect_heel_toe_events(self.pose_data)
        # *********************************************************************
        if self.rotated_pose_data is None:
             self.logger.warning("EventDetector did not return rotated pose data. Using original pose data for step length calculation (which might be less accurate).")
             self.rotated_pose_data = self.pose_data # Fallback, but calculation expects rotated
        else:
             self.logger.info("Event detection complete. Stored rotated pose data.")
        return self.events


    def compute_gait_parameters(self):
        # MODIFIED: Pass rotated_pose_data
        if self.events is None or self.pose_data is None or self.frame_rate is None:
             raise ValueError("Cannot compute parameters, prerequisites (events, pose_data, frame_rate) not met.")
        if self.rotated_pose_data is None:
            # Fallback added in detect_events, but log warning if it's using non-rotated data
            self.logger.warning("Computing parameters using non-rotated pose data for step length.")
            effective_pose_data_for_sl = self.pose_data
        else:
            effective_pose_data_for_sl = self.rotated_pose_data

        self.logger.info("Computing gait parameters...")
        gait_computer = GaitParameters()
        # *** Pass rotated data to the modified compute_parameters method ***
        self.gait_params = gait_computer.compute_parameters(
            self.events,
            self.pose_data,             # Original data (for non-SL params if needed)
            effective_pose_data_for_sl, # Rotated data (specifically for revised SL)
            self.frame_rate,
            save_path=None
        )
        self.logger.info("Gait parameters computation complete.")
        return self.gait_params

    def compute_forward_displacement(self):
        # ... (no changes needed here) ...
        """Computes and stores the average forward displacement signal."""
        if self.pose_data is None:
            raise ValueError("Cannot compute forward displacement, pose_data is not loaded.")
        self.logger.info("Computing forward displacement...")
        try:
            # Ensure columns exist using MultiIndex access if needed
            left_toe_col = ('left_foot_index', 'z') if isinstance(self.pose_data.columns, pd.MultiIndex) else 'left_foot_index_z'
            right_toe_col = ('right_foot_index', 'z') if isinstance(self.pose_data.columns, pd.MultiIndex) else 'right_foot_index_z'
            sacrum_col = ('sacrum', 'z') if isinstance(self.pose_data.columns, pd.MultiIndex) else 'sacrum_z'

            if not all(col in self.pose_data.columns for col in [left_toe_col, right_toe_col, sacrum_col]):
                 missing = [col for col in [left_toe_col, right_toe_col, sacrum_col] if col not in self.pose_data.columns]
                 raise KeyError(f"Required columns missing for forward displacement: {missing}")

            left_toe = self.pose_data[left_toe_col]
            right_toe = self.pose_data[right_toe_col]
            sacrum = self.pose_data[sacrum_col]

        except KeyError as e:
            self.logger.error(f"KeyError accessing columns for forward displacement: {e}")
            raise ValueError("Required markers not found in pose data for forward displacement.") from e

        left_forward = left_toe - sacrum
        right_forward = right_toe - sacrum
        # Use nanmean to handle potential NaNs robustly
        self.forward_displacement = np.nanmean(np.vstack([left_forward.to_numpy(), right_forward.to_numpy()]), axis=0)
        # Check if result is valid
        if self.forward_displacement is None or np.all(np.isnan(self.forward_displacement)):
             raise ValueError("Forward displacement calculation resulted in NaNs or None.")
        self.logger.info("Forward displacement computation complete.")
        return self.forward_displacement

    def detect_freezes(self):
        # ... (no changes needed here) ...
        """Detects Freezing of Gait (FoG) events."""
        if self.forward_displacement is None:
             try:
                 self.compute_forward_displacement()
             except ValueError as e:
                  self.logger.error(f"Cannot detect freezes because forward displacement computation failed: {e}")
                  self.fog_events = [] # Return empty list if cannot compute prerequisite
                  return self.fog_events

        if self.frame_rate is None:
            raise ValueError("Cannot detect freezes, frame_rate is not set.")

        self.logger.info("Starting FoG detection...")
        freezing_config = self.config.get("freezing", {})
        fd = FreezingDetector(
            frame_rate=self.frame_rate,
            window_size_sec=freezing_config.get("window_size_sec", 2.0),
            step_size_sec=freezing_config.get("step_size_sec", 0.5),
            velocity_threshold=freezing_config.get("velocity_threshold", 0.1),
            fi_threshold=freezing_config.get("fi_threshold", 2.0)
        )
        self.fog_events = fd.detect_freezes(self.forward_displacement)
        self.logger.info(f"FoG detection complete. Found {len(self.fog_events)} potential FoG events.")
        return self.fog_events


# MODIFIED process_gait_file function
def process_gait_file(input_file, config, details_save_path, plots_save_path): # Removed event_plots_dir
    """
    Process a single file using the integrated gait pipeline. Saves detailed results
    and generates plots (including the former validation plot). Returns summary stats.

    Parameters:
      input_file (str): Path to the input file (CSV or video).
      config (dict): Configuration settings for the gait analysis.
      details_save_path (str): Directory to save detailed per-step CSV.
      plots_save_path (str): Directory to save all kinematic analysis plots.

    Returns:
      tuple: (summary_df, skipped_file)
          - summary_df (pd.DataFrame): Summary statistics, or None if failed/skipped.
          - skipped_file (str): Base name if skipped due to error, else None.
    """
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    logger.info(f"Initiating gait processing for: {base_name}")
    skipped_file = None
    pipeline = None

    try:
        # Step 1: YOLO-based cropping (same as before)
        if input_file.lower().endswith((".mp4", ".mov")):
            # ... (keep cropping logic as is) ...
            cropper_config = config.get("yolo_cropper", {})
            cropper = YOLOCropper(confidence_threshold=cropper_config.get("confidence_threshold", 0.5))
            base, ext = os.path.splitext(input_file)
            cropped_video_dir = os.path.join(os.path.dirname(config["pose_estimator"]["tracked_video_dir"]), "cropped_videos")
            os.makedirs(cropped_video_dir, exist_ok=True)
            cropped_video_path = os.path.join(cropped_video_dir, f"{base_name}_cropped{ext}")

            if os.path.exists(cropped_video_path):
                logger.info(f"Using existing cropped video: {cropped_video_path}")
                input_file_for_pipeline = cropped_video_path
            else:
                logger.info(f"Cropping video: {input_file}")
                cropped_file, cropped_size = cropper.crop_video(
                    input_video_path=input_file,
                    output_video_path=cropped_video_path
                )
                if cropped_file is None:
                    logger.warning(f"Cropping failed for {input_file}. Skipping file.")
                    return None, base_name
                input_file_for_pipeline = cropped_file
                if cropped_size and len(cropped_size) == 2:
                     config["pose_estimator"]["image_dimensions"] = cropped_size
                     logger.info(f"Updated image dimensions in config: {cropped_size}")
                logger.info(f"Cropped video saved to: {cropped_video_path}")
        else:
             input_file_for_pipeline = input_file

        # Step 2: Instantiate and run the gait pipeline
        pipeline = GaitPipeline(input_path=input_file_for_pipeline, config=config)
        pipeline.load_input()
        pipeline.preprocess()
        pipeline.detect_events() # This now stores self.rotated_pose_data

        # Step 3: Automatically save the "Event Detection Validation" plot
        plot_config = config.get("gait_plotting", {}) # Get plotting config section
        # Add a specific flag for this plot if desired, or tie it to generate_kinematic_plots
        if plot_config.get("plot_event_signals", True): # New flag example
             logger.info("Generating event detection signals plot...")
             try:
                 # Recalculate signals needed (as before)
                 left_toe_col = ('left_foot_index', 'z') if isinstance(pipeline.pose_data.columns, pd.MultiIndex) else 'left_foot_index_z'
                 right_toe_col = ('right_foot_index', 'z') if isinstance(pipeline.pose_data.columns, pd.MultiIndex) else 'right_foot_index_z'
                 sacrum_col = ('sacrum', 'z') if isinstance(pipeline.pose_data.columns, pd.MultiIndex) else 'sacrum_z'
                 toe_left_signal = pipeline.pose_data[left_toe_col] - pipeline.pose_data[sacrum_col]
                 toe_right_signal = pipeline.pose_data[right_toe_col] - pipeline.pose_data[sacrum_col]

                 if not (toe_left_signal.empty or toe_right_signal.empty or
                    (toe_left_signal.nunique() <= 1 and toe_right_signal.nunique() <= 1)):

                     fs = pipeline.frame_rate
                     filter_config = config.get("event_detection", {}).get("filter", {})
                     cutoff = filter_config.get("cutoff", 3.0)
                     order = filter_config.get("order", 4)
                     filtered_left = butter_lowpass_filter(toe_left_signal.to_numpy(), cutoff=cutoff, fs=fs, order=order)
                     filtered_right = butter_lowpass_filter(toe_right_signal.to_numpy(), cutoff=cutoff, fs=fs, order=order)

                     all_forward_movement = {"TO_left": filtered_left, "TO_right": filtered_right}
                     peaks_left, valleys_left = detect_extremas(filtered_left)
                     peaks_right, valleys_right = detect_extremas(filtered_right)
                     all_extrema_data = {
                         "TO_left": {"peaks": peaks_left / fs, "valleys": valleys_left / fs},
                         "TO_right": {"peaks": peaks_right / fs, "valleys": valleys_right / fs}
                     }

                     # Call plotting function to GENERATE and SAVE the plot
                     fig = plot_combined_extremas_and_toe(
                         all_forward_movement,
                         all_extrema_data,
                         fs,
                         input_file, # Use original name for context
                         output_dir=plots_save_path, # Save to kinematic plots dir
                         save_plot=True, # Explicitly save
                         # show_plot=False # Parameter removed from function
                     )
                     if fig:
                          plt.close(fig) # Close figure after saving
                          logger.info(f"Saved event detection signals plot for {base_name}.")
                     else:
                          logger.warning(f"Failed to generate event detection signals plot for {base_name}.")
                 else:
                      logger.warning("Forward displacement signals for event plot are empty or constant. Skipping plot generation.")

             except Exception as viz_err:
                  logger.error(f"Error generating event detection signals plot for {base_name}: {viz_err}", exc_info=True)
                  # Continue processing even if this plot fails

        # REMOVED: Block calling prompt_visualisation and checking 'approved'

        # Step 4: Compute gait parameters, FoG, and save detailed data
        gait_parameters = pipeline.compute_gait_parameters() # Calls method which now uses rotated data for SL

        # --- FoG Detection ---
        try:
            pipeline.detect_freezes()
        except Exception as fog_err:
            logger.error(f"Error during FoG detection for {base_name}: {fog_err}", exc_info=True)
            # Continue processing, FoG results will be empty/zero


        # Save detailed gait parameters
        if gait_parameters is not None and not gait_parameters.empty:
            detailed_csv_path = os.path.join(details_save_path, f"{base_name}_gait_details.csv")
            try:
                save_csv(gait_parameters, detailed_csv_path)
                logger.info(f"Saved detailed gait parameters to: {detailed_csv_path}")
            except Exception as save_err:
                logger.error(f"Failed to save detailed gait parameters for {base_name}: {save_err}", exc_info=True)
        else:
            logger.warning(f"No detailed gait parameters computed or DataFrame is empty for {base_name}. Cannot save details or generate summary/plots.")
            # If no parameters, likely cannot proceed
            return None, base_name


        # Step 5: Generate Kinematic Summary Plots (Non-interactive)
        # plot_config is already defined above
        if plot_config.get("generate_kinematic_plots", True):
             # ... (keep kinematic plot generation logic as is) ...
             logger.info(f"Generating kinematic plots for {base_name}...")
             plot_kwargs = {'gait_df': gait_parameters, 'output_dir': plots_save_path, 'video_name': base_name}
             plot_error_count = 0

             # Plot Time Series
             if plot_config.get("plot_timeseries", True):
                 try:
                     params_ts = plot_config.get("timeseries_params", ['stride_duration', 'step_length', 'gait_speed', 'swing', 'cadence'])
                     plot_parameter_timeseries(parameters=params_ts, **plot_kwargs)
                 except Exception as e:
                     logger.error(f"Failed to generate timeseries plot: {e}", exc_info=False)
                     plot_error_count += 1

             # Plot Distributions
             if plot_config.get("plot_distributions", True):
                 try:
                     params_dist = plot_config.get("distribution_params", ['stride_duration', 'step_length', 'step_duration', 'swing'])
                     plot_parameter_distributions(parameters=params_dist, **plot_kwargs)
                 except Exception as e:
                     logger.error(f"Failed to generate distribution plot: {e}", exc_info=False)
                     plot_error_count += 1

             # Plot Left/Right Comparison
             if plot_config.get("plot_left_right_comparison", True):
                 try:
                     params_lr = plot_config.get("left_right_params", ['step_duration', 'step_length', 'swing', 'stance', 'initial_double_support'])
                     plot_left_right_comparison(parameters=params_lr, **plot_kwargs)
                 except Exception as e:
                     logger.error(f"Failed to generate left/right comparison plot: {e}", exc_info=False)
                     plot_error_count += 1

            # Plot FoG Visualization
             if plot_config.get("plot_fog_visualization", True) and pipeline.fog_events is not None:
                 if pipeline.forward_displacement is not None and len(pipeline.forward_displacement) > 0:
                     try:
                         plot_fog_visualization(
                             forward_disp=pipeline.forward_displacement,
                             fog_events=pipeline.fog_events,
                             frame_rate=pipeline.frame_rate,
                             output_dir=plots_save_path,
                             video_name=base_name
                         )
                     except Exception as e:
                         logger.error(f"Failed to generate FoG visualization plot: {e}", exc_info=False)
                         plot_error_count += 1
                 else:
                      logger.warning("Skipping FoG plot generation: Forward displacement data not available or empty.")

             if plot_error_count == 0:
                 logger.info(f"Successfully generated kinematic plots for {base_name}.")
             else:
                  logger.warning(f"Generated plots for {base_name} with {plot_error_count} errors.")


        # Step 6: Compute Enhanced Summary Statistics
        logger.info(f"Calculating summary statistics for {base_name}...")
        stats_to_compute = ['mean', 'median', 'std', 'count']
        summary_stats = gait_parameters.agg(stats_to_compute, numeric_only=True) # Should work now if step length is numeric

        if isinstance(summary_stats.index, pd.MultiIndex):
            summary_stats = summary_stats.unstack(level=[1, 2])
        else:
            summary_stats = summary_stats.unstack()

        summary_stats.columns = ["_".join(map(str, col)).strip('_') for col in summary_stats.columns.values]
        summary_df = pd.DataFrame(summary_stats) # This transposes automatically if needed

        # Check if DataFrame is empty before proceeding
        if summary_df.empty and not summary_stats.empty: # If unstack resulted in empty DF but stats exist
             summary_df = pd.DataFrame([summary_stats]) # Handle case where agg gives Series

        # Add FoG summary
        fog_count = len(pipeline.fog_events) if pipeline.fog_events is not None else 0
        fog_total_duration = sum(event["duration_sec"] for event in pipeline.fog_events) if pipeline.fog_events else 0.0
        summary_df["fog_count"] = fog_count
        summary_df["fog_total_duration_sec"] = fog_total_duration

        # Add video name identifier
        summary_df.insert(0, "video_name", base_name)

        # Reorder columns (optional)
        # cols = ["video_name"] + sorted([col for col in summary_df.columns if col != "video_name"])
        # summary_df = summary_df[cols]

        logger.info(f"Successfully processed {base_name}; gait analysis complete.")
        return summary_df, None # Return summary, no skipped file due to validation

    except Exception as e:
        logger.error(f"An error occurred during processing of {base_name}: {e}", exc_info=True)
        return None, base_name # Return None summary, mark as skipped due to error