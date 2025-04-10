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
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s") # Changed level to INFO
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning) # Ignore potential runtime warnings from stats/plots

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
            # Ensure fallback exists if it's the chosen one
            os.makedirs(fallback, exist_ok=True)
            return fallback


def main():
    # Get the project root (two levels above main.py)
    project_root = get_project_root()

    # Load config.json
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    try:
        config = load_config(config_path)
    except FileNotFoundError:
        logger.error(f"Configuration file not found at: {config_path}")
        return
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from configuration file: {config_path}")
        return

    # Determine which analysis module to run from config.
    analysis_module = config.get("analysis_module", "gait")
    logger.info("Selected analysis module: %s", analysis_module)

    # Resolve the data and output directories:
    data_dir = get_external_folder("Data", project_root, config.get("data_dir", "data")) # Add default fallback
    output_dir = get_external_folder("Output", project_root, config.get("output_dir", "output")) # Add default fallback

    # Define main kinematic features output directory
    kinematic_features_dir = os.path.join(output_dir, "kinematic_features")

    # Update paths for output subdirectories from config using the resolved output_dir
    # Use kinematic_features_dir as the base for gait-related outputs
    config["gait_parameters"] = config.get("gait_parameters", {}) # Ensure gait_parameters exists
    config["gait_parameters"]["base_save_path"] = kinematic_features_dir # Store the base path
    config["gait_parameters"]["details_save_path"] = os.path.join(kinematic_features_dir, "per_video_details")
    config["gait_parameters"]["plots_save_path"] = os.path.join(kinematic_features_dir, "plots")
    config["gait_parameters"]["summary_save_path"] = kinematic_features_dir # Master summary goes here

    # Pose estimator paths remain relative to the main output_dir
    config["pose_estimator"] = config.get("pose_estimator", {}) # Ensure pose_estimator exists
    config["pose_estimator"]["tracked_csv_dir"] = os.path.join(output_dir, config["pose_estimator"].get("tracked_csv_dir", "tracked_data/csv"))
    config["pose_estimator"]["tracked_video_dir"] = os.path.join(output_dir, config["pose_estimator"].get("tracked_video_dir", "tracked_data/video"))

    # Event detection plot path (interactive validation plot)
    config["event_detection"] = config.get("event_detection", {}) # Ensure event_detection exists
    config["event_detection"]["plots_dir"] = os.path.join(output_dir, config["event_detection"].get("plots_dir", "plots/event_validation")) # Keep separate from kinematic plots

    # Ensure that the needed directories exist
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(config["gait_parameters"]["base_save_path"], exist_ok=True)
    os.makedirs(config["gait_parameters"]["details_save_path"], exist_ok=True)
    os.makedirs(config["gait_parameters"]["plots_save_path"], exist_ok=True)
    os.makedirs(config["pose_estimator"]["tracked_csv_dir"], exist_ok=True)
    os.makedirs(config["pose_estimator"]["tracked_video_dir"], exist_ok=True)
    os.makedirs(config["event_detection"]["plots_dir"], exist_ok=True)

    # Gather input files from data_dir (CSV, MP4, MOV, etc.)
    # Exclude any file name containing "_cropped" or "_skipped" so they won't be processed again
    input_files = []
    skipped_or_cropped_suffixes = ("_cropped", "_skipped")
    for ext in ("*.csv", "*.mp4", "*.MP4", "*.mov", "*.MOV"):
        for path in glob.glob(os.path.join(data_dir, ext)):
            base_name = os.path.splitext(os.path.basename(path))[0]
            if not any(base_name.endswith(suffix) for suffix in skipped_or_cropped_suffixes):
                input_files.append(path)

    if not input_files:
        logger.error("No valid input files (excluding '_cropped', '_skipped') found in %s", data_dir)
        return

    logger.info("Found %d files to process in %s", len(input_files), data_dir)

    # Lists to store summary DataFrames and skipped file names
    all_summaries = []
    skipped_files_log = [] # Renamed to avoid conflict with skipped_file variable

    # Process each file based on the selected analysis module
    for input_file in input_files:
        logger.info(f"--- Processing file: {os.path.basename(input_file)} ---")
        summary_df = None # Initialize summary_df for the current file
        skipped_file = None # Initialize skipped_file status for the current file

        if analysis_module == "gait":
            try:
                # Ensure event_plots_dir is REMOVED from this call:
                summary_df, skipped_file = process_gait_file(
                    input_file,
                    config,
                    details_save_path=config["gait_parameters"]["details_save_path"],
                    plots_save_path=config["gait_parameters"]["plots_save_path"]
                    # NO event_plots_dir argument here anymore!
                )
            except Exception as e:
                logger.error(f"Error processing gait file {os.path.basename(input_file)}: {e}", exc_info=True)
                skipped_file = os.path.basename(input_file)
                
        elif analysis_module == "tremor":
            # Placeholder for tremor analysis - ensure it returns a similar structure
            try:
                from modules.tremor_pipeline import run_tremor_analysis
                tremor_features = run_tremor_analysis(input_file, config) # Assuming this returns a dict or DataFrame
                video_name = os.path.splitext(os.path.basename(input_file))[0]

                if isinstance(tremor_features, pd.DataFrame):
                    # Assuming tremor_features is already the summary DataFrame
                    summary_df = tremor_features.copy()
                    if "video_name" not in summary_df.columns:
                         summary_df.insert(0, "video_name", video_name)
                elif isinstance(tremor_features, dict):
                     # Convert dict to DataFrame row
                     summary_df = pd.DataFrame([tremor_features])
                     summary_df.insert(0, "video_name", video_name)
                else:
                    logger.warning(f"Unexpected output type from tremor analysis for {video_name}: {type(tremor_features)}")
                    summary_df = pd.DataFrame([{"video_name": video_name, "tremor_processing_status": "failed"}])

                # Ensure column order if possible
                if summary_df is not None and "video_name" in summary_df.columns:
                    cols = ["video_name"] + [col for col in summary_df.columns if col != "video_name"]
                    summary_df = summary_df[cols]

            except ImportError:
                logger.error("Tremor analysis module not found.")
                skipped_file = os.path.basename(input_file)
            except Exception as e:
                logger.error(f"Error processing tremor file {os.path.basename(input_file)}: {e}", exc_info=True)
                skipped_file = os.path.basename(input_file) # Mark as skipped due to error

        else:
            logger.error("Invalid analysis module specified in config: %s", analysis_module)
            # Optionally skip the file or stop execution
            skipped_file = os.path.basename(input_file)

        # Append results or skipped files
        if summary_df is not None:
            all_summaries.append(summary_df)
            logger.info(f"Successfully generated summary for: {os.path.basename(input_file)}")
        elif skipped_file is not None:
            skipped_files_log.append(skipped_file)
            logger.warning(f"File skipped: {skipped_file}")
        else:
             # This case should ideally not happen if logic above is correct
             logger.warning(f"File processed but no summary generated and not marked as skipped: {os.path.basename(input_file)}")


    # After processing all files, combine summaries and save master summary CSV.
    if all_summaries:
        try:
            master_summary = pd.concat(all_summaries, ignore_index=True)
            master_summary_csv_path = os.path.join(config["gait_parameters"]["summary_save_path"], "all_summary.csv")
            save_csv(master_summary, master_summary_csv_path)
            logger.info("Master summary saved to %s", master_summary_csv_path)
        except Exception as e:
            logger.error(f"Failed to concatenate or save master summary: {e}", exc_info=True)
    else:
        logger.warning("No summaries were generated for any file.")

    if skipped_files_log:
        logger.warning("--- Summary of Skipped Files ---")
        for f in skipped_files_log:
            logger.warning("  - %s", f)
        logger.warning("--- End of Skipped Files ---")

    logger.info("--- Pipeline execution finished ---")


if __name__ == "__main__":
    main()