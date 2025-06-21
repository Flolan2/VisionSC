#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 26 20:16:41 2025

@author: Lange_L
"""

# src/kinematic_tracker_standalone.py

import os
import glob
import json
import logging
import argparse
import sys

# Ensure 'modules' can be imported correctly
try:
    from modules.pose_estimation import PoseEstimator
except ImportError as e:
    print(f"ImportError: {e}. Could not import PoseEstimator from modules.")
    print("Please ensure you are running this script from your code project's root directory (e.g., `python src/kinematic_tracker_standalone.py`),")
    print("or that the `src` directory is in your PYTHONPATH, or run as a module (e.g., `python -m src.kinematic_tracker_standalone`).")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
logger = logging.getLogger(__name__)

def get_code_project_root(script_file_path):
    """
    Determines the root directory of the code project.
    Assumes the script is located within a 'src' directory,
    and 'src' is a direct child of the code project's root directory.
    """
    current_path = os.path.abspath(script_file_path)
    while True:
        parent_of_current_dir, current_dir_name = os.path.split(os.path.dirname(current_path))
        if current_dir_name == "src":
            return parent_of_current_dir
        if not current_dir_name:
            logger.error("Could not determine code project root: 'src' directory not found in the path.")
            return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(script_file_path)), "..", ".."))
        current_path = os.path.dirname(current_path)

def load_main_config(config_arg_path, code_project_root_path):
    """
    Loads the main configuration file.
    Tries CLI arg path, then default config.json in code_project_root/src/.
    Returns the full config dictionary.
    """
    if config_arg_path:
        config_file_to_load = os.path.abspath(config_arg_path)
    else:
        src_dir_path = os.path.join(code_project_root_path, "src")
        config_file_to_load = os.path.join(src_dir_path, "config.json")

    try:
        with open(config_file_to_load, "r") as f:
            full_config = json.load(f)
        logger.info(f"Successfully loaded main configuration from: {config_file_to_load}")
        return full_config
    except FileNotFoundError:
        logger.error(f"Main configuration file not found at: {config_file_to_load}. Cannot proceed without configuration.")
        sys.exit(1)
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from: {config_file_to_load}. Cannot proceed.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error loading config from {config_file_to_load}: {e}. Cannot proceed.")
        sys.exit(1)
    return {} # Should not be reached if sys.exit() is called

def run_kinematic_tracking(args):
    """
    Main function to find cropped videos, run pose estimation, and save outputs.
    """
    code_project_root = get_code_project_root(__file__)
    external_io_base_dir = os.path.abspath(os.path.join(code_project_root, ".."))

    # Define input and output directories based on the external structure
    cropped_videos_input_dir = os.path.join(external_io_base_dir, "output", "cropped_videos")
    target_tracked_csv_output_dir = os.path.join(external_io_base_dir, "output", "tracked_data", "csv")
    target_tracked_video_output_dir = os.path.join(external_io_base_dir, "output", "tracked_data", "video")

    # Create output directories if they don't exist (PoseEstimator also does this, but good for clarity)
    os.makedirs(target_tracked_csv_output_dir, exist_ok=True)
    os.makedirs(target_tracked_video_output_dir, exist_ok=True)

    logger.info(f"Input cropped videos directory: {cropped_videos_input_dir}")
    logger.info(f"Tracked CSVs will be saved to: {target_tracked_csv_output_dir}")
    logger.info(f"Tracked videos will be saved to: {target_tracked_video_output_dir}")

    full_config = load_main_config(args.config_path, code_project_root)
    
    # PoseEstimator will use make_video and make_csv from the config directly.
    # We pass the full config to its constructor.
    try:
        pose_estimator_instance = PoseEstimator(config=full_config)
    except Exception as e:
        logger.error(f"Failed to initialize PoseEstimator: {e}")
        logger.error("This might be due to model file issues or MediaPipe initialization problems.")
        sys.exit(1)
        
    video_extensions_to_scan = ("*.mp4", "*.MP4", "*.mov", "*.MOV")
    input_video_files = []
    for ext in video_extensions_to_scan:
        input_video_files.extend(glob.glob(os.path.join(cropped_videos_input_dir, ext)))

    if not input_video_files:
        logger.info(f"No video files found in '{cropped_videos_input_dir}' with extensions {video_extensions_to_scan}.")
        return

    logger.info(f"Found {len(input_video_files)} video files to process for pose estimation.")

    # Suffix to identify videos that are MediaPipe outputs
    mp_tracked_suffix = "_MPtracked"

    for video_file_path in input_video_files:
        video_basename_full = os.path.basename(video_file_path)
        video_basename_no_ext, video_ext = os.path.splitext(video_basename_full)

        # Construct expected output paths for checking/deletion
        expected_csv_path = os.path.join(target_tracked_csv_output_dir, f"{video_basename_no_ext}{mp_tracked_suffix}.csv")
        expected_metadata_path = expected_csv_path.replace(".csv", "_metadata.json")
        expected_video_path = os.path.join(target_tracked_video_output_dir, f"{video_basename_no_ext}{mp_tracked_suffix}{video_ext if video_ext.lower() == '.mp4' else '.mp4'}") # MP usually saves as .mp4

        if args.overwrite:
            if os.path.exists(expected_csv_path):
                logger.info(f"Overwrite requested. Deleting existing CSV: {expected_csv_path}")
                os.remove(expected_csv_path)
            if os.path.exists(expected_metadata_path):
                logger.info(f"Overwrite requested. Deleting existing metadata: {expected_metadata_path}")
                os.remove(expected_metadata_path)
            if os.path.exists(expected_video_path):
                logger.info(f"Overwrite requested. Deleting existing video: {expected_video_path}")
                os.remove(expected_video_path)
        
        # PoseEstimator.process_video will check for existing CSV and skip if found (unless deleted by overwrite)
        logger.info(f"Starting pose estimation for: {video_file_path}")
        try:
            # Pass the target output directories directly to override PoseEstimator's defaults
            marker_df, fps = pose_estimator_instance.process_video(
                video_path=video_file_path,
                tracked_csv_dir=target_tracked_csv_output_dir,
                tracked_video_dir=target_tracked_video_output_dir
            )
            if marker_df is not None and fps is not None:
                logger.info(f"Successfully processed (or loaded existing) for: {video_file_path}. FPS: {fps}")
            else:
                logger.warning(f"Pose estimation did not return data for {video_file_path}.")
        except Exception as e:
            logger.error(f"An error occurred during pose estimation for {video_file_path}: {e}", exc_info=True)

    logger.info("Kinematic tracking (pose estimation) process completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Standalone script for kinematic tracking (pose estimation) using MediaPipe."
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Optional: Path to the main 'config.json' file. If not provided, "
             "it defaults to 'your_code_project_root/src/config.json'."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, existing tracked CSVs and videos will be deleted and re-processed."
    )
    
    cli_args = parser.parse_args()
    run_kinematic_tracking(cli_args)