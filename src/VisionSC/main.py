import os
import glob
import json
import logging
import warnings
import pandas as pd

from modules.gait.my_utils.helpers import save_csv

# Import gait pipeline controller
from modules.gait_pipeline import process_gait_file

# Configure logging and warnings
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)


def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)


def get_project_root():
    """
    Returns the absolute path two levels above this file.
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def get_external_folder(external_name, project_root, fallback_relative):
    """
    For the "Output" folder: always use the external folder (one level above the project)
    with the given name. If it doesn't exist, create it.

    For other folders (e.g., "Data"), use the external folder if it exists; otherwise,
    fall back to the relative folder provided.
    """
    parent_dir = os.path.abspath(os.path.join(project_root, ".."))
    external_path = os.path.join(parent_dir, external_name)
    if external_name == "Output":
        if not os.path.isdir(external_path):
            logger.info("External %s folder not found. Creating folder: %s", external_name, external_path)
            os.makedirs(external_path, exist_ok=True)
        else:
            logger.info("Using external %s folder: %s", external_name, external_path)
        return external_path
    else:
        if os.path.isdir(external_path):
            logger.info("Using external %s folder: %s", external_name, external_path)
            return external_path
        else:
            fallback = os.path.join(project_root, fallback_relative)
            logger.info("External %s folder not found. Using internal folder: %s", external_name, fallback)
            return fallback


def main():
    # Get the project root (two levels above main.py)
    project_root = get_project_root()

    # Load config.json
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    config = load_config(config_path)

    # Determine which analysis module to run from config.
    analysis_module = config.get("analysis_module", "gait")
    logger.info("Selected analysis module: %s", analysis_module)

    # Resolve the data and output directories:
    data_dir = get_external_folder("Data", project_root, config["data_dir"])
    output_dir = get_external_folder("Output", project_root, config["output_dir"])

    # Update paths for output subdirectories from config using the resolved output_dir
    # Rename gait_parameters folder to "kinematic_features"
    config["gait_parameters"]["save_path"] = os.path.join(output_dir, "kinematic_features")
    config["pose_estimator"]["tracked_csv_dir"] = os.path.join(output_dir, config["pose_estimator"]["tracked_csv_dir"])
    config["pose_estimator"]["tracked_video_dir"] = os.path.join(output_dir, config["pose_estimator"]["tracked_video_dir"])
    config["event_detection"]["plots_dir"] = os.path.join(output_dir, config["event_detection"]["plots_dir"])

    # Ensure that the needed directories exist
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(config["gait_parameters"]["save_path"], exist_ok=True)
    os.makedirs(config["pose_estimator"]["tracked_csv_dir"], exist_ok=True)
    os.makedirs(config["pose_estimator"]["tracked_video_dir"], exist_ok=True)
    os.makedirs(config["event_detection"]["plots_dir"], exist_ok=True)

    # Gather input files from data_dir (CSV, MP4, MOV, etc.)
    # Exclude any file name containing "_cropped" so they won't be processed again
    input_files = []
    for ext in ("*.csv", "*.mp4", "*.MP4", "*.mov", "*.MOV"):
        for path in glob.glob(os.path.join(data_dir, ext)):
            if "_cropped" not in os.path.basename(path):
                input_files.append(path)

    if not input_files:
        logger.error("No valid input files (excluding '_cropped') found in %s", data_dir)
        return

    logger.info("Found %d files to process in %s", len(input_files), data_dir)

    # Lists to store summary DataFrames and skipped file names
    all_summaries = []
    skipped_files = []

    # Process each file based on the selected analysis module
    for input_file in input_files:
        if analysis_module == "gait":
            summary_df, skipped_file = process_gait_file(input_file, config, config["gait_parameters"]["save_path"])
        elif analysis_module == "tremor":
            from modules.tremor_pipeline import run_tremor_analysis
            tremor_features = run_tremor_analysis(input_file, config)
            video_name = os.path.splitext(os.path.basename(input_file))[0]
            if isinstance(tremor_features, pd.DataFrame):
                tremor_features["video_name"] = video_name
                summary_df = tremor_features.reset_index(drop=True)
                if "video_name" in summary_df.columns:
                    columns = ["video_name"] + [col for col in summary_df.columns if col != "video_name"]
                    summary_df = summary_df[columns]
            else:
                summary_df = pd.DataFrame({
                    "video_name": [video_name],
                    "dominant_tremor_frequency": [tremor_features.get("dominant_freq", None)],
                    "tremor_amplitude": [tremor_features.get("tremor_amplitude", None)],
                    "frame_rate": [tremor_features.get("frame_rate", None)],
                    "n_frames": [tremor_features.get("n_frames", None)]
                })
            skipped_file = None
        else:
            logger.error("Invalid analysis module specified: %s", analysis_module)
            continue

        if summary_df is not None:
            all_summaries.append(summary_df)
        elif skipped_file is not None:
            skipped_files.append(skipped_file)

    # After processing all files, combine summaries and save master summary CSV.
    if all_summaries:
        master_summary = pd.concat(all_summaries, ignore_index=True)
        master_summary_csv_path = os.path.join(config["gait_parameters"]["save_path"], "all_summary.csv")
        save_csv(master_summary, master_summary_csv_path)
        logger.info("Master summary saved to %s", master_summary_csv_path)
    else:
        logger.info("No summaries were generated.")

    if skipped_files:
        logger.info("The following files were skipped:")
        for f in skipped_files:
            logger.info("  %s", f)


if __name__ == "__main__":
    main()
