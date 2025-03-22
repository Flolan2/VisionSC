#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tremor_integration.py
Modified version: Excessive logging output removed
"""

import os
import logging
from typing import Optional, Union
import numpy as np
import pandas as pd

# Import the robust FPS extraction helper.
from modules.helpers import get_robust_fps

# Import the utility functions that have been moved to tremor_utils.
from .tremor.tremor_utils import load_tracking_csv

logger = logging.getLogger(__name__)

# --- Monkey Patch Start ---
try:
    from .tremor import tremor_features
    def dummy_assign_hand_time_periods(pc):
        # Minimal logging to indicate monkey patching without verbose output.
        logger.debug("Skipping active time period assignment (monkey patched).")
        return pc
    tremor_features.assign_hand_time_periods = dummy_assign_hand_time_periods
except Exception as e:
    logger.error("Failed to monkey patch assign_hand_time_periods: %s", e)
# --- Monkey Patch End ---


def run_tracking(video_path: str, csv_path: str, config: dict) -> None:
    logger.info("### ENTERING TRACKING")
    from .tremor.tremor_tracking import load_models, track_video

    hands, pose = load_models(
        min_hand_detection_confidence=config.get("min_hand_detection_confidence", 0.5),
        min_tracking_confidence=config.get("min_tracking_confidence", 0.7)
    )
    logger.info("Tracking video: %s", video_path)
    
    csv_output_folder = config["pose_estimator"]["tracked_csv_dir"]
    video_output_folder = config["pose_estimator"]["tracked_video_dir"]
    
    os.makedirs(csv_output_folder, exist_ok=True)
    
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
    
    video_name = os.path.basename(video_path).split('.')[0]
    tracked_video_filename = f"{video_name}_tracked.mp4"
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
    logger.info("### STARTING PREPROCESSING")
    from .tremor.tremor_preprocessing import construct_data
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
        # Removed debug_print_shoulder_markers to reduce output.
    return pc


def run_marker_extraction(pc):
    logger.info("### ENTERING MARKER EXTRACTION")
    from .tremor.tremor_marker_extraction import (
        extract_intention_tremor,
        extract_postural_tremor,
        extract_proximal_tremor,
        extract_distal_tremor,
        extract_fingers_tremor
    )
    pc = extract_intention_tremor(pc)
    pc = extract_postural_tremor(pc)
    pc = extract_proximal_tremor(pc)
    pc = extract_distal_tremor(pc)
    pc = extract_fingers_tremor(pc)
    logger.info("Marker extraction complete.")
    # Removed debug printing of marker data.
    return pc


def run_postprocessing(pc):
    logger.info("### ENTERING POSTPROCESSING")
    from .tremor.tremor_pca_features import assign_hand_time_periods
    pc = assign_hand_time_periods(pc)
    logger.info("Postprocessing complete.")
    # Removed detailed postprocessing data printout.
    return pc


def run_feature_extraction(pc) -> Union[pd.DataFrame, dict]:
    logger.info("### ENTERING FEATURE EXTRACTION")
    from .tremor.tremor_pca_features import (
        extract_proximal_arm_tremor_features,
        extract_distal_arm_tremor_features,
        extract_fingers_tremor_features
    )
    
    try:
        proximal_arm_features = extract_proximal_arm_tremor_features(pc, plot=False, save_plots=False)
        if proximal_arm_features is None or proximal_arm_features.empty:
            logger.warning("Proximal arm tremor features are empty for all patients. Using empty DataFrame.")
            proximal_arm_features = pd.DataFrame(index=pc.get_patient_ids())
    except Exception as e:
        logger.error("Error extracting proximal arm tremor features: %s", e)
        proximal_arm_features = pd.DataFrame(index=pc.get_patient_ids())
    
    try:
        distal_arm_features = extract_distal_arm_tremor_features(pc, plot=False, save_plots=False)
        if distal_arm_features is None or distal_arm_features.empty:
            logger.warning("Distal arm tremor features are empty for all patients. Using empty DataFrame.")
            distal_arm_features = pd.DataFrame(index=pc.get_patient_ids())
    except Exception as e:
        logger.error("Error extracting distal arm tremor features: %s", e)
        distal_arm_features = pd.DataFrame(index=pc.get_patient_ids())
    
    try:
        fingers_features = extract_fingers_tremor_features(pc, plot=False, save_plots=False)
        if fingers_features is None or fingers_features.empty:
            logger.warning("Fingers tremor features are empty for all patients. Using empty DataFrame.")
            fingers_features = pd.DataFrame(index=pc.get_patient_ids())
    except Exception as e:
        logger.error("Error extracting fingers tremor features: %s", e)
        fingers_features = pd.DataFrame(index=pc.get_patient_ids())
    
    try:
        combined_features = pd.concat([proximal_arm_features, distal_arm_features, fingers_features], axis=1)
    except Exception as e:
        logger.error("Error combining tremor features: %s", e)
        combined_features = pd.DataFrame(index=pc.get_patient_ids())
    
    if combined_features.columns.duplicated().any():
        logger.warning("Duplicate columns found in tremor features. Removing duplicates.")
        combined_features = combined_features.loc[:, ~combined_features.columns.duplicated()]
    
    logger.info("Feature extraction complete.")
    # Removed logging of the final tremor features columns.
    return combined_features


def run_tremor_analysis(video_path: str, config: dict) -> Optional[Union[pd.DataFrame, dict]]:
    csv_output_folder = config["pose_estimator"]["tracked_csv_dir"]
    video_name = os.path.basename(video_path).split('.')[0]
    csv_path = os.path.join(csv_output_folder, f"{video_name}_MPtracked.csv")
    
    # Check for an existing valid tracking CSV.
    track_data = None
    if os.path.exists(csv_path):
        track_data = load_tracking_csv(csv_path, video_path)
        if track_data is not None:
            logger.info("Existing valid tracking CSV found for %s, skipping tracking.", video_path)
    
    if track_data is None:
        run_tracking(video_path, csv_path, config)
    
    logger.info("### ENTERING PREPROCESSING")
    track_data = load_tracking_csv(csv_path, video_path)
    if track_data is None:
        logger.error("Failed to load valid tracking data for %s", video_path)
        return None

    logger.info("### ENTERING FRAME RATE EXTRACTION")
    try:
        frame_rate = get_robust_fps(video_path, tolerance=0.1)
        logger.info("Extracted frame rate: %s FPS", frame_rate)
    except Exception as e:
        logger.warning("FPS extraction failed for %s: %s", video_path, e)
        frame_rate = config.get("frame_rate", 30.0)
        logger.info("Using fallback frame rate: %s FPS", frame_rate)

    pc = run_preprocessing(csv_path, frame_rate, config)
    if pc is None:
        return None

    from .tremor.tremor_normalization import normalize_pose_data
    pc = normalize_pose_data(
        pc,
        global_scaling_factor=config.get("global_scaling_factor", 1.0),
        use_local=True,
        use_zscore=True
    )
    
    pc = run_marker_extraction(pc)
    pc = run_postprocessing(pc)
    tremor_features = run_feature_extraction(pc)

    if isinstance(tremor_features, dict):
        tremor_features['frame_rate'] = frame_rate
    elif isinstance(tremor_features, pd.DataFrame):
        tremor_features['frame_rate'] = frame_rate

    logger.info("### PIPELINE COMPLETE")
    
    try:
        from .tremor.tremor_utils import executive_summary
        summary_text = executive_summary(tremor_features)
        
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
        output_dir = os.path.join(base_dir, "Output", "kinematic_features")
        os.makedirs(output_dir, exist_ok=True)
        summary_path = os.path.join(output_dir, f"{video_name}_executive_summary.txt")
        
        with open(summary_path, "w") as f:
            f.write(summary_text)
        logger.info("Executive summary written to %s", summary_path)
    except Exception as e:
        logger.error("Failed to create executive summary: %s", e)

    return tremor_features
