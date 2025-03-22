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
import matplotlib.pyplot as plt
import argparse

# Import YOLO cropper for video processing
from modules.yolo_cropper import YOLOCropper

# Helpers for pose data loading and frame rate retrieval
from modules.gait.my_utils.helpers import load_csv, get_frame_rate

# Optional: import visualization helpers
from modules.gait.my_utils.plotting import (
    butter_lowpass_filter,
    detect_extremas,
    plot_combined_extremas_and_toe,
)
from modules.gait.my_utils.prompt_visualisation import prompt_visualisation

# Import additional modules used by the gait pipeline
from modules.pose_estimation import PoseEstimator
from modules.gait.gait_preprocessing import Preprocessor
from modules.gait.gait_event_detection import EventDetector
from modules.gait.gait_parameters_computation import GaitParameters

# Configure logging and warnings
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)


# Combined GaitPipeline class (integrated directly into this controller)
class GaitPipeline:
    def __init__(self, input_path, config, save_parameters_path):
        self.input_path = input_path
        self.config = config
        self.save_parameters_path = save_parameters_path
        self.pose_data = None
        self.frame_rate = None
        self.events = None
        self.gait_params = None
        self.fog_events = None  # Freezing of Gait events
        
        # Initialize a logger for this class
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        if not self.logger.hasHandlers():
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

    def load_input(self):
        if self.input_path.endswith(".csv"):
            self.pose_data = load_csv(file_path=self.input_path)
            self.frame_rate = get_frame_rate(file_path=self.input_path)
        elif self.input_path.endswith((".mp4", ".MP4", ".mov", ".MOV")):
            pose_estimator = PoseEstimator(config=self.config)
            self.pose_data, self.frame_rate = pose_estimator.process_video(video_path=self.input_path)
            if self.pose_data is not None:
                self.pose_data = self.pose_data.apply(pd.to_numeric, errors="coerce")
        else:
            raise ValueError("Unsupported input format. Use .mp4/.mov for videos or .csv for spreadsheets.")
        return self.pose_data

    def preprocess(self):
        preprocessor = Preprocessor(pose_data=self.pose_data)
        self.pose_data = preprocessor.preprocess(window_size=self.config["preprocessing"]["median_filter_window"])
        return self.pose_data

    def detect_events(self):
        detector = EventDetector(**self.config["event_detection"], input_path=self.input_path, frame_rate=self.frame_rate)
        self.events = detector.detect_heel_toe_events(self.pose_data)
        return self.events

    def compute_gait_parameters(self):
        gait_params = GaitParameters()
        self.gait_params = gait_params.compute_parameters(
            self.events, self.pose_data, self.frame_rate, save_path=self.save_parameters_path
        )
        # Recompute step lengths with the correct pairing:
        left_step_length = GaitParameters.compute_step_length(
            self.events, self.pose_data, self.frame_rate, side="left", other_side="right"
        )
        right_step_length = GaitParameters.compute_step_length(
            self.events, self.pose_data, self.frame_rate, side="right", other_side="left"
        )
        self.gait_params[("left", "step_length")] = left_step_length
        self.gait_params[("right", "step_length")] = right_step_length

        self.logger.debug("Recomputed left step length (first few values): %s", left_step_length.head())
        self.logger.debug("Recomputed right step length (first few values): %s", right_step_length.head())
        return self.gait_params

    def compute_forward_displacement(self):
        try:
            left_toe = self.pose_data[("left_foot_index", "z")]
            right_toe = self.pose_data[("right_foot_index", "z")]
            sacrum = self.pose_data[("sacrum", "z")]
        except KeyError as e:
            raise ValueError("Required markers not found in pose data. Ensure pose_data includes 'left_foot_index', 'right_foot_index', and 'sacrum'.") from e

        left_forward = left_toe - sacrum
        right_forward = right_toe - sacrum
        forward_disp = (left_forward + right_forward) / 2.0
        return forward_disp.to_numpy()

    def detect_freezes(self):
        from modules.gait.gait_freezing_detector import FreezingDetector  # Import here to avoid circular dependencies
        forward_disp = self.compute_forward_displacement()
        freezing_config = self.config.get("freezing", {})
        velocity_threshold = freezing_config.get("velocity_threshold", 0.05)
        fi_threshold = freezing_config.get("fi_threshold", 2.0)
        window_size_sec = freezing_config.get("window_size_sec", 2.0)
        step_size_sec = freezing_config.get("step_size_sec", 0.5)
        fd = FreezingDetector(
            frame_rate=self.frame_rate,
            window_size_sec=window_size_sec,
            step_size_sec=step_size_sec,
            velocity_threshold=velocity_threshold,
            fi_threshold=fi_threshold
        )
        self.fog_events = fd.detect_freezes(forward_disp)
        return self.fog_events


# Controller function that processes a single file using the combined pipeline
def process_gait_file(input_file, config, output_dir):
    """
    Process a single file using the integrated gait pipeline.

    Parameters:
      input_file (str): Path to the input file (CSV or video).
      config (dict): Configuration settings for the gait analysis.
      output_dir (str): Output directory for saving visualizations or other files.

    Returns:
      tuple: (summary_df, skipped_file)
          - summary_df (pd.DataFrame): DataFrame containing computed gait parameters, or None if processing was skipped.
          - skipped_file (str): File name if the file was skipped, else None.
    """
    skipped_file = None

    # Step 1: YOLO-based cropping for video files
    if input_file.lower().endswith((".mp4", ".mov")):
        cropper = YOLOCropper(confidence_threshold=config.get("yolo_confidence_threshold", 0.5))
        base, ext = os.path.splitext(input_file)
        cropped_video_path = f"{base}_cropped{ext}"
        if os.path.exists(cropped_video_path):
            logger.info("Cropped video already exists: %s", cropped_video_path)
            input_file = cropped_video_path
        else:
            cropped_file, cropped_size = cropper.crop_video(
                input_video_path=input_file,
                output_video_path=cropped_video_path
            )
            input_file = cropped_file
            config["pose_estimator"]["image_dimensions"] = cropped_size

    # Step 2: Instantiate and run the gait pipeline
    pipeline = GaitPipeline(input_path=input_file, config=config, save_parameters_path=None)
    pose_data = pipeline.load_input()
    if pose_data is None:
        logger.info("Skipping %s due to loading issues.", input_file)
        return None, input_file

    pipeline.preprocess()
    pipeline.detect_events()

    # Optional: Visualization if enabled in the configuration
    if config.get("visualize", False):
        toe_left_signal = pose_data[("left_foot_index", "z")] - pose_data[("sacrum", "z")]
        toe_right_signal = pose_data[("right_foot_index", "z")] - pose_data[("sacrum", "z")]

        all_forward_movement = {
            "TO_left": toe_left_signal.to_numpy(),
            "TO_right": toe_right_signal.to_numpy()
        }
        fs = pipeline.frame_rate
        filtered_left = butter_lowpass_filter(all_forward_movement["TO_left"], cutoff=3, fs=fs)
        filtered_right = butter_lowpass_filter(all_forward_movement["TO_right"], cutoff=3, fs=fs)
        all_forward_movement["TO_left"] = filtered_left
        all_forward_movement["TO_right"] = filtered_right

        peaks_left, valleys_left = detect_extremas(filtered_left)
        peaks_right, valleys_right = detect_extremas(filtered_right)
        all_extrema_data = {
            "TO_left": {"peaks": peaks_left / fs, "valleys": valleys_left / fs},
            "TO_right": {"peaks": peaks_right / fs, "valleys": valleys_right / fs}
        }

        if (toe_left_signal.empty or toe_right_signal.empty or
            (toe_left_signal.nunique() <= 1 and toe_right_signal.nunique() <= 1)):
            logger.warning("Forward displacement signals are empty or constant. Skipping visualization.")
        else:
            fig = plot_combined_extremas_and_toe(
                all_forward_movement,
                all_extrema_data,
                fs,
                input_file,
                output_dir=None,
                show_plot=True
            )
            approved, new_file = prompt_visualisation(fig, input_file, config["event_detection"]["plots_dir"])
            if not approved:
                plt.close(fig)
                return None, new_file
            plt.close(fig)

    # Step 3: Compute gait parameters and detect freeze events
    gait_parameters = pipeline.compute_gait_parameters()
    fog_events = pipeline.detect_freezes()
    if fog_events:
        fog_count = len(fog_events)
        fog_total_duration = sum(event["duration_sec"] for event in fog_events)
    else:
        fog_count = 0
        fog_total_duration = 0.0

    median_summary = gait_parameters.median(numeric_only=True)
    if isinstance(median_summary.index, pd.MultiIndex):
        median_summary.index = ["_".join(map(str, tup)).strip() for tup in median_summary.index.values]
    summary_df = pd.DataFrame(median_summary).T

    video_name = os.path.splitext(os.path.basename(input_file))[0]
    summary_df["video_name"] = video_name
    summary_df["fog_count"] = fog_count
    summary_df["fog_total_duration_sec"] = fog_total_duration

    columns = ["video_name"] + [col for col in summary_df.columns if col != "video_name"]
    summary_df = summary_df[columns]

    logger.info("Processed %s; gait analysis complete.", input_file)
    return summary_df, None

