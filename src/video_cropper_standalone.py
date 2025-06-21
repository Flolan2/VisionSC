# src/video_cropper_standalone.py

import os
import glob
import json
import logging
import argparse
import sys
import warnings

# --- Suppress FutureWarning from yolov5 ---
# This ignores the specific "torch.cuda.amp.autocast(args...)" deprecation warning
# that floods the console when running YOLOv5 object detection on each frame.
# The warning originates from the library code, not our code.
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda.amp.autocast")
# --- End of Suppression Block ---


# Ensure 'modules' can be imported.
try:
    from modules.yolo_cropper import YOLOCropper
except ImportError as e:
    print(f"ImportError: {e}. Could not import YOLOCropper from modules.")
    print("Please ensure you are running this script from the project_root directory (e.g., `python src/video_cropper_standalone.py`),")
    print("or that the `src` directory is in your PYTHONPATH, or run as a module (e.g., `python -m src.video_cropper_standalone`).")
    sys.exit(1)


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
logger = logging.getLogger(__name__)

def get_code_project_root(script_file_path):
    """
    Determines the root directory of the code project.
    Assumes the script is located within a 'src' directory,
    and 'src' is a direct child of the code project's root directory.
    Example: /path/to/parent_dir/my_code_project/src/script.py -> /path/to/parent_dir/my_code_project
    """
    current_path = os.path.abspath(script_file_path)
    # Traverse upwards from the script's directory until 'src' is found
    while True:
        # Get the directory containing current_path, then split that to get its parent and its own name
        parent_of_current_dir, current_dir_name = os.path.split(os.path.dirname(current_path))
        if current_dir_name == "src":
            # parent_of_current_dir is now the directory containing 'src', which is our code_project_root
            return parent_of_current_dir
        if not current_dir_name: # Reached the root of the filesystem
            logger.error("Could not determine code project root: 'src' directory not found in the path.")
            # Fallback: assume code project root is two levels above script if 'src' isn't found.
            # This is a less robust fallback.
            return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(script_file_path)), "..", ".."))
        current_path = os.path.dirname(current_path) # Move one level up from current_path for next iteration

def load_yolo_configuration(config_arg_path, code_project_root_path):
    """
    Loads YOLO cropper specific configuration.
    Tries CLI arg path, then default config.json in code_project_root/src/.
    """
    default_yolo_params = {
        "confidence_threshold": 0.5,
        "margin": 80
    }

    if config_arg_path:
        config_file_to_load = os.path.abspath(config_arg_path)
    else:
        # Default location: code_project_root/src/config.json
        src_dir_path = os.path.join(code_project_root_path, "src")
        config_file_to_load = os.path.join(src_dir_path, "config.json")

    try:
        with open(config_file_to_load, "r") as f:
            full_config = json.load(f)
        yolo_params_from_file = full_config.get("yolo_cropper", {})
        
        final_yolo_params = default_yolo_params.copy()
        final_yolo_params.update(yolo_params_from_file)

        logger.info(f"Successfully loaded YOLO configuration from: {config_file_to_load}")
        return final_yolo_params
    except FileNotFoundError:
        logger.warning(f"Main configuration file not found at: {config_file_to_load}. Using default YOLO parameters.")
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from: {config_file_to_load}. Using default YOLO parameters.")
    except Exception as e:
        logger.error(f"Unexpected error loading config from {config_file_to_load}: {e}. Using default YOLO parameters.")
    return default_yolo_params


def run_video_cropping(args):
    """
    Main function to find videos, crop them, and save them.
    Assumes 'data' and 'output' folders are siblings to the code project root.
    """
    # This identifies the root directory of your *code project* (e.g., the directory containing "src")
    code_project_root = get_code_project_root(__file__) # Pass path to this script

    # The base directory for external data and output folders is the PARENT of the code_project_root
    external_io_base_dir = os.path.abspath(os.path.join(code_project_root, ".."))

    data_dir = os.path.join(external_io_base_dir, "data")
    # The root for this script's output (e.g., parent_dir_of_code_project/output/)
    script_output_base_dir = os.path.join(external_io_base_dir, "output")
    cropped_videos_output_dir = os.path.join(script_output_base_dir, "cropped_videos")

    if not os.path.isdir(data_dir):
        logger.error(f"Data directory does not exist: {data_dir}")
        logger.error("Please ensure your 'data' folder is located one level above your code project directory.")
        sys.exit(1)
    
    os.makedirs(cropped_videos_output_dir, exist_ok=True)
    logger.info(f"Input data directory: {data_dir}")
    logger.info(f"Cropped videos will be saved to: {cropped_videos_output_dir}")

    # Configuration file is expected inside the code_project_root/src/
    yolo_config_params = load_yolo_configuration(args.config_path, code_project_root)
    
    try:
        yolo_cropper_instance = YOLOCropper(
            confidence_threshold=yolo_config_params.get("confidence_threshold", 0.5)
        )
    except Exception as e:
        logger.error(f"Failed to initialize YOLOCropper: {e}")
        logger.error("This might be due to missing dependencies like 'torch' or 'ultralytics', or model download issues.")
        sys.exit(1)
        
    video_extensions_to_scan = ("*.mp4", "*.MP4", "*.mov", "*.MOV")
    input_video_files = []
    for ext in video_extensions_to_scan:
        input_video_files.extend(glob.glob(os.path.join(data_dir, ext)))

    if not input_video_files:
        logger.info(f"No video files found in '{data_dir}' with extensions {video_extensions_to_scan}.")
        return

    logger.info(f"Found {len(input_video_files)} video files to process.")

    excluded_suffixes = ("_cropped", "_skipped", "_MPtracked")

    for video_file_path in input_video_files:
        video_basename_full = os.path.basename(video_file_path)
        video_basename_no_ext, video_ext = os.path.splitext(video_basename_full)

        if any(video_basename_no_ext.endswith(suffix) for suffix in excluded_suffixes):
            logger.info(f"Skipping file with excluded suffix: {video_basename_full}")
            continue

        target_cropped_video_path = os.path.join(cropped_videos_output_dir, f"{video_basename_no_ext}_cropped{video_ext}")

        if os.path.exists(target_cropped_video_path) and not args.overwrite:
            logger.info(f"Cropped video already exists: {target_cropped_video_path}. Use --overwrite to replace.")
            continue

        logger.info(f"Starting cropping for: {video_file_path}")
        try:
            cropped_file_result, cropped_dimensions = yolo_cropper_instance.crop_video(
                input_video_path=video_file_path,
                output_video_path=target_cropped_video_path,
                margin=yolo_config_params.get("margin", 80)
            )
            if cropped_file_result:
                logger.info(f"Successfully cropped video saved to: {cropped_file_result}")
                if cropped_dimensions:
                    logger.info(f"Cropped video dimensions: {cropped_dimensions}")
            else:
                logger.warning(f"Cropping returned no file for {video_file_path}. Check YOLOCropper logs if any.")
        except Exception as e:
            logger.error(f"An error occurred while cropping {video_file_path}: {e}", exc_info=True)

    logger.info("Video cropping process completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Standalone script to crop videos using YOLO person detection."
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
        help="If set, existing cropped videos will be overwritten."
    )
    
    cli_args = parser.parse_args()
    run_video_cropping(cli_args)