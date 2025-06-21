# src/gait_processor_standalone.py

import os
import glob
import json
import logging
import argparse
import sys
import pandas as pd

# Ensure project modules can be imported
try:
    # Corrected imports assuming 'gait' and 'my_utils' are packages directly under 'src'
    from modules.gait.gait_preprocessing import Preprocessor
    from modules.gait.gait_event_detection import EventDetector
    
    # If you decide to include event signal plotting here from my_utils:
    # from my_utils.plotting import plot_combined_extremas_and_toe
    # from my_utils.helpers import butter_lowpass_filter, detect_extremas # if needed for plotting
except ImportError as e:
    print(f"ImportError: {e}. Could not import gait processing modules.")
    print("Ensure __init__.py files are present in 'src/gait/' and 'src/my_utils/' (if used).")
    print("Also ensure you are running this script from your code project's root directory (e.g., `python src/gait_processor_standalone.py`),")
    print("or that `src`'s parent directory is in PYTHONPATH, or run as a module (`python -m src.gait_processor_standalone`).")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
logger = logging.getLogger(__name__)

def get_code_project_root(script_file_path):
    # Assumes script is in src/, and src/ is child of code_project_root
    current_path = os.path.abspath(script_file_path) 
    # script_file_path is .../Online/src/gait_processor_standalone.py
    # os.path.dirname(current_path) is .../Online/src
    # os.path.dirname(os.path.dirname(current_path)) is .../Online (code_project_root)
    return os.path.dirname(os.path.dirname(current_path))


def load_main_config(config_arg_path, code_project_root_path):
    if config_arg_path:
        config_file_to_load = os.path.abspath(config_arg_path)
    else:
        # Config is in src/config.json
        src_dir_path = os.path.join(code_project_root_path, "src")
        config_file_to_load = os.path.join(src_dir_path, "config.json")
    try:
        with open(config_file_to_load, "r") as f:
            full_config = json.load(f)
        logger.info(f"Successfully loaded main configuration from: {config_file_to_load}")
        return full_config
    except Exception as e:
        logger.error(f"Error loading config from {config_file_to_load}: {e}. Cannot proceed.", exc_info=True)
        sys.exit(1)
    return {}

def save_dataframe_to_csv(df, output_path):
    """Helper to save DataFrame and ensure directory exists."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Successfully saved data to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save DataFrame to {output_path}: {e}", exc_info=True)

def run_gait_processing(args):
    code_project_root = get_code_project_root(__file__)
    external_io_base_dir = os.path.abspath(os.path.join(code_project_root, ".."))

    tracked_csv_input_dir = os.path.join(external_io_base_dir, "output", "tracked_data", "csv")
    events_output_dir = os.path.join(external_io_base_dir, "output", "gait_events")
    proc_rotated_output_dir = os.path.join(external_io_base_dir, "output", "processed_rotated_data")

    os.makedirs(events_output_dir, exist_ok=True)
    os.makedirs(proc_rotated_output_dir, exist_ok=True)

    logger.info(f"Input tracked CSVs directory: {tracked_csv_input_dir}")
    logger.info(f"Output gait events directory: {events_output_dir}")
    logger.info(f"Output processed/rotated data directory: {proc_rotated_output_dir}")

    full_config = load_main_config(args.config_path, code_project_root)
    preprocessing_config = full_config.get("preprocessing", {})
    event_detection_config = full_config.get("event_detection", {})

    input_csv_files = glob.glob(os.path.join(tracked_csv_input_dir, "*_MPtracked.csv"))

    if not input_csv_files:
        logger.info(f"No tracked CSV files found in '{tracked_csv_input_dir}'.")
        return

    logger.info(f"Found {len(input_csv_files)} tracked CSV files to process.")

    for csv_file_path in input_csv_files:
        base_name_with_suffix = os.path.basename(csv_file_path)
        original_video_basename = base_name_with_suffix.replace("_MPtracked.csv", "")
        logger.info(f"--- Processing: {original_video_basename} (from {base_name_with_suffix}) ---")

        metadata_path = csv_file_path.replace(".csv", "_metadata.json")
        fps = 30.0 
        if os.path.isfile(metadata_path):
            try:
                with open(metadata_path, "r") as f: metadata = json.load(f)
                fps = float(metadata.get("fps", fps))
                if fps <= 0:
                    logger.warning(f"Invalid FPS ({fps}) in metadata {metadata_path}. Using default {30.0} FPS.")
                    fps = 30.0
                logger.info(f"Using FPS: {fps:.2f} from metadata file.")
            except Exception as e:
                logger.warning(f"Error reading FPS from {metadata_path}: {e}. Using default {fps:.2f} FPS.")
        else:
            logger.warning(f"Metadata file not found: {metadata_path}. Using default {fps:.2f} FPS.")

        try:
            pose_data_df = pd.read_csv(csv_file_path, header=[0, 1]) 
            logger.info(f"Loaded pose data from {csv_file_path}, shape: {pose_data_df.shape}")
        except Exception as e:
            logger.error(f"Failed to load pose data from {csv_file_path}: {e}", exc_info=True)
            continue 

        try:
            logger.info("Initializing Preprocessor...")
            preprocessor = Preprocessor(pose_data=pose_data_df.copy()) 
            logger.info("Running preprocessing...")
            median_filter_window = preprocessing_config.get("median_filter_window", 11) 
            preprocessed_data_df = preprocessor.preprocess(window_size=median_filter_window)
            logger.info("Preprocessing complete.")
        except Exception as e:
            logger.error(f"Error during preprocessing for {original_video_basename}: {e}", exc_info=True)
            continue

        try:
            logger.info("Initializing EventDetector...")
            detector = EventDetector(
                input_path=csv_file_path, 
                algorithm=event_detection_config.get("algorithm", "zeni"),
                frame_rate=fps,
                window_size=event_detection_config.get("rotation_window_size", 100),
                step_size=event_detection_config.get("rotation_step_size", 50),
                config=full_config 
            )
            logger.info("Running event detection...")
            events_df, rotated_pose_data_df = detector.detect_heel_toe_events(preprocessed_data_df)
            
            if events_df is not None and not events_df.empty:
                logger.info(f"Event detection complete. Found {events_df.notna().sum().sum()} event instances.")
            else:
                logger.warning("Event detection resulted in empty or None events DataFrame.")
            
            if rotated_pose_data_df is None or rotated_pose_data_df.empty :
                 logger.warning("Event detection did not return rotated pose data. This will affect step length calculation.")
                 rotated_pose_data_df = preprocessed_data_df 
        except Exception as e:
            logger.error(f"Error during event detection for {original_video_basename}: {e}", exc_info=True)
            # Ensure dataframes are initialized even on error to prevent issues in save block
            events_df = pd.DataFrame() 
            rotated_pose_data_df = pd.DataFrame() # Or preprocessed_data_df if preferred fallback
            # continue # Option to skip saving if event detection fails catastrophically

        # Save outputs even if they might be empty (to signify processing attempt or partial success)
        events_output_path = os.path.join(events_output_dir, f"{original_video_basename}_events.csv")
        save_dataframe_to_csv(events_df, events_output_path)

        # It's crucial to save rotated_pose_data_df as it's used for step length.
        # If it's empty due to an error, an empty file will be saved.
        proc_rotated_output_path = os.path.join(proc_rotated_output_dir, f"{original_video_basename}_proc_rotated.csv")
        save_dataframe_to_csv(rotated_pose_data_df, proc_rotated_output_path)
        
        logger.info(f"--- Finished processing for: {original_video_basename} ---")

    logger.info("Gait processing (preprocessing & event detection) standalone script finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone script for gait data preprocessing and event detection.")
    parser.add_argument(
        "--config_path", type=str, default=None,
        help="Optional: Path to the main 'config.json' file. Defaults to 'code_project_root/src/config.json'."
    )
    cli_args = parser.parse_args()
    run_gait_processing(cli_args)