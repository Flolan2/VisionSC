#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
poet_integration.py
Created on Mon Mar  3 21:58:39 2025
Author: Lange_L

This module runs the full PoET pipeline (tracking → preprocessing → marker extraction → postprocessing → feature extraction)
on a single video file and returns tremor metrics.

Note: The preprocessing step now preserves the multi-index keypoint format 
(e.g.. columns with (keypoint, coordinate) tuples) so that the downstream feature extraction code works as expected.
This version only supports the MultiIndex CSV format.
"""

import os
import logging
from typing import Optional, Union
import numpy as np
import pandas as pd

# Import the robust FPS extraction helper from your gait module's helpers.
from my_utils.helpers import get_robust_fps

logger = logging.getLogger(__name__)


def debug_print_shoulder_markers(df: pd.DataFrame, step: str):
    """
    Logs the column names and first two rows of any columns containing 'shoulder'
    from the provided DataFrame.
    """
    shoulder_cols = [col for col in df.columns if 'shoulder' in col[0].lower()]
    if shoulder_cols:
        logger.info("Step %s: Found shoulder markers: %s", step, shoulder_cols)
        logger.info("Step %s: Data sample for shoulder markers:\n%s", step, df.loc[:, shoulder_cols].head(2))
    else:
        logger.info("Step %s: No shoulder markers found.", step)


# --- Monkey Patch Start ---
# Disable active time period assignment by replacing it with a dummy function.
try:
    from .PoET import poet_features
    def dummy_assign_hand_time_periods(pc):
        logger.info("Skipping active time period assignment (monkey patched).")
        return pc
    poet_features.assign_hand_time_periods = dummy_assign_hand_time_periods
except Exception as e:
    logger.error("Failed to monkey patch assign_hand_time_periods: %s", e)
# --- Monkey Patch End ---


def load_tracking_csv(csv_path: str, video_path: str) -> Optional[pd.DataFrame]:
    """
    Attempts to load and validate the tracking CSV file.
    Returns a valid DataFrame if successful, or None if the CSV is empty, invalid,
    or not in the expected MultiIndex format.
    """
    try:
        data = pd.read_csv(csv_path, header=[0, 1], index_col=0)
        if data.empty:
            logger.info("Tracking CSV exists for %s but is empty; will re-run tracking.", video_path)
            return None
        if not isinstance(data.columns, pd.MultiIndex):
            logger.error("Tracking CSV for %s is not in the expected MultiIndex format.", video_path)
            return None

        # Debug: print shoulder markers in the tracking CSV.
        debug_print_shoulder_markers(data, "Tracking CSV")
        return data
    except Exception as e:
        logger.warning("Error reading tracking CSV for %s: %s; will re-run tracking.", video_path, e)
        return None


def run_tracking(video_path: str, csv_path: str, config: dict) -> None:
    """
    Runs the tracking step if a valid CSV is not already present.
    The tracking CSV is saved to config["pose_estimator"]["tracked_csv_dir"].
    The tracked video is initially saved there too, but then moved to config["pose_estimator"]["tracked_video_dir"].
    """
    logger.info("### ENTERING TRACKING")
    from .PoET.poet_tracking import load_models, track_video
    hands, pose = load_models(
        min_hand_detection_confidence=config.get("min_hand_detection_confidence", 0.5),
        min_tracking_confidence=config.get("min_tracking_confidence", 0.7)
    )
    logger.info("Tracking video: %s", video_path)
    
    # Retrieve the output folders from config.
    csv_output_folder = config["pose_estimator"]["tracked_csv_dir"]
    video_output_folder = config["pose_estimator"]["tracked_video_dir"]
    
    # Ensure the CSV output folder exists.
    os.makedirs(csv_output_folder, exist_ok=True)
    
    # Call track_video. We assume that track_video writes both CSV and tracked video files into csv_output_folder.
    track_video(
        video=video_path,
        pose=pose,
        hands=hands,
        output_folder=csv_output_folder,
        make_csv=True,
        make_video=True,
        plot=True,
        world_coords=True
    )
    
    # After tracking, move the tracked video from the CSV folder to the tracked video folder.
    video_name = os.path.basename(video_path).split('.')[0]
    tracked_video_filename = video_name + '_tracked.mp4'
    source_video_path = os.path.join(csv_output_folder, tracked_video_filename)
    target_video_path = os.path.join(video_output_folder, tracked_video_filename)
    
    if os.path.exists(source_video_path):
        os.makedirs(video_output_folder, exist_ok=True)
        try:
            os.rename(source_video_path, target_video_path)
            logger.info("Moved tracked video from %s to %s", source_video_path, target_video_path)
        except Exception as e:
            logger.error("Failed to move tracked video from %s to %s: %s", source_video_path, target_video_path, e)
    else:
        logger.warning("Tracked video file %s not found in %s", tracked_video_filename, csv_output_folder)


def run_preprocessing(csv_path: str, frame_rate: float, config: dict):
    """
    Constructs the PatientCollection using the tracking CSV and returns it.
    """
    logger.info("### STARTING PREPROCESSING")
    from .PoET.poet_preprocessing import construct_data
    pc = construct_data(
        csv_files=[csv_path],
        fs=[frame_rate],
        labels=[None],
        scaling_factor=config.get("scaling_factor", 1),
        verbose=True
    )
    if pc is None:
        logger.error("Preprocessing failed for %s", csv_path)
    else:
        logger.info("Preprocessing complete.")
        # Debug: For each patient, print shoulder markers from the stored pose_estimation 
        for patient in pc.patients:
            df = patient.pose_estimation
            if hasattr(patient, "structural_features"):
                df = patient.structural_features
            debug_print_shoulder_markers(df, f"Preprocessing for patient {patient.patient_id}")
    return pc


def run_marker_extraction(pc):
    """
    Replaces the old kinematics step by extracting marker data for tremor analysis.
    Calls functions to extract intention, postural, proximal, distal, and finger tremor marker data,
    storing the results in dedicated attributes on each patient.
    """
    logger.info("### ENTERING MARKER EXTRACTION")
    from .PoET.poet_kinematics import (
        extract_intention_tremor,
        extract_postural_tremor,
        extract_proximal_tremor,
        extract_distal_tremor,
        extract_fingers_tremor  # Added to extract finger markers
    )
    pc = extract_intention_tremor(pc)
    pc = extract_postural_tremor(pc)
    pc = extract_proximal_tremor(pc)
    pc = extract_distal_tremor(pc)
    pc = extract_fingers_tremor(pc)  # Now finger tremor markers are extracted
    logger.info("Marker extraction complete.")
    # Debug: Check for shoulder markers in the newly stored proximal tremor features.
    for patient in pc.patients:
        if hasattr(patient, "proximal_tremor_features"):
            debug_print_shoulder_markers(patient.proximal_tremor_features, f"Marker Extraction for patient {patient.patient_id}")
    return pc


def run_postprocessing(pc):
    """
    Assigns hand time periods in the processed data.
    Now imports the assign_hand_time_periods function from poet_features.
    """
    logger.info("### ENTERING POSTPROCESSING")
    from .PoET.poet_features import assign_hand_time_periods
    pc = assign_hand_time_periods(pc)
    logger.info("Postprocessing complete.")
    # Debug: Check for shoulder markers after postprocessing.
    for patient in pc.patients:
        df = patient.pose_estimation
        if hasattr(patient, "structural_features"):
            df = patient.structural_features
        debug_print_shoulder_markers(df, f"Postprocessing for patient {patient.patient_id}")
    return pc


def run_feature_extraction(pc) -> Union[pd.DataFrame, dict]:
    """
    Extracts tremor features for proximal arm, distal arm, and fingers,
    then combines them into a single output.
    Now imports the feature extraction functions from poet_features.
    This version is made more robust so that if any extraction fails or returns
    an empty DataFrame, it is replaced with an empty DataFrame.
    """
    logger.info("### ENTERING FEATURE EXTRACTION")
    from .PoET.poet_features import (
        extract_proximal_arm_tremor_features,
        extract_distal_arm_tremor_features,
        extract_fingers_tremor_features
    )
    
    # Extract proximal arm tremor features.
    try:
        proximal_arm_features = extract_proximal_arm_tremor_features(pc, plot=False, save_plots=False)
        if proximal_arm_features is None or proximal_arm_features.empty:
            logger.warning("Proximal arm tremor features are empty for all patients. Using empty DataFrame.")
            proximal_arm_features = pd.DataFrame(index=pc.get_patient_ids())
    except Exception as e:
        logger.error("Error extracting proximal arm tremor features: %s", e)
        proximal_arm_features = pd.DataFrame(index=pc.get_patient_ids())
    
    # Extract distal arm tremor features.
    try:
        distal_arm_features = extract_distal_arm_tremor_features(pc, plot=False, save_plots=False)
        if distal_arm_features is None or distal_arm_features.empty:
            logger.warning("Distal arm tremor features are empty for all patients. Using empty DataFrame.")
            distal_arm_features = pd.DataFrame(index=pc.get_patient_ids())
    except Exception as e:
        logger.error("Error extracting distal arm tremor features: %s", e)
        distal_arm_features = pd.DataFrame(index=pc.get_patient_ids())
    
    # Extract fingers tremor features.
    try:
        fingers_features = extract_fingers_tremor_features(pc, plot=False, save_plots=False)
        if fingers_features is None or fingers_features.empty:
            logger.warning("Fingers tremor features are empty for all patients. Using empty DataFrame.")
            fingers_features = pd.DataFrame(index=pc.get_patient_ids())
    except Exception as e:
        logger.error("Error extracting fingers tremor features: %s", e)
        fingers_features = pd.DataFrame(index=pc.get_patient_ids())
    
    # Combine the features side-by-side.
    try:
        combined_features = pd.concat([proximal_arm_features, distal_arm_features, fingers_features], axis=1)
    except Exception as e:
        logger.error("Error combining tremor features: %s", e)
        combined_features = pd.DataFrame(index=pc.get_patient_ids())
    
    # Check for duplicate columns and remove duplicates if necessary.
    if combined_features.columns.duplicated().any():
        logger.warning("Duplicate columns found in tremor features. Removing duplicates.")
        combined_features = combined_features.loc[:, ~combined_features.columns.duplicated()]
    
    logger.info("Feature extraction complete.")
    # Debug: Print final feature DataFrame columns
    logger.info("Final tremor features columns: %s", combined_features.columns.tolist())
    return combined_features


def run_poet_analysis(video_path: str, config: dict) -> Optional[Union[pd.DataFrame, dict]]:
    """
    Runs the full PoET pipeline on a single video file and returns tremor metrics.
    This version only supports CSV tracking data with a MultiIndex header.
    
    Updated to use marker extraction (instead of the removed kinematic step) and the new feature extraction.
    """
    # Use the tracked CSV directory from the main script.
    csv_output_folder = config["pose_estimator"]["tracked_csv_dir"]
    
    # Construct the expected CSV path for tracking data.
    video_name = os.path.basename(video_path).split('.')[0]
    csv_path = os.path.join(csv_output_folder, video_name + '_MPtracked.csv')
    
    # Check if a valid tracking CSV already exists.
    track_data = None
    if os.path.exists(csv_path):
        track_data = load_tracking_csv(csv_path, video_path)
        if track_data is not None:
            logger.info("Existing valid tracking CSV found for %s, skipping tracking.", video_path)
    
    # If tracking CSV is not available or invalid, run the tracking step.
    if track_data is None:
        run_tracking(video_path, csv_path, config)
    
    # After tracking (or if skipped), load the tracking CSV.
    logger.info("### ENTERING PREPROCESSING")
    track_data = load_tracking_csv(csv_path, video_path)
    if track_data is None:
        logger.error("Failed to load valid tracking data for %s", video_path)
        return None

    # Frame Rate Extraction: use the robust FPS helper.
    logger.info("### ENTERING FRAME RATE EXTRACTION")
    try:
        frame_rate = get_robust_fps(video_path, tolerance=0.1)
        logger.info("Extracted frame rate: %s FPS", frame_rate)
    except Exception as e:
        logger.warning("FPS extraction failed for %s: %s", video_path, str(e))
        frame_rate = config.get("frame_rate", 30.0)
        logger.info("Using fallback frame rate: %s FPS", frame_rate)

    # Preprocessing.
    pc = run_preprocessing(csv_path, frame_rate, config)
    if pc is None:
        return None

    # Marker Extraction (replacing kinematics).
    pc = run_marker_extraction(pc)
    
    # Postprocessing.
    pc = run_postprocessing(pc)

    # Feature extraction.
    tremor_features = run_feature_extraction(pc)

    # Optionally, include the detected frame rate in the output.
    if isinstance(tremor_features, dict):
        tremor_features['frame_rate'] = frame_rate
    elif isinstance(tremor_features, pd.DataFrame):
        tremor_features['frame_rate'] = frame_rate

    logger.info("### PIPELINE COMPLETE")
    return tremor_features
