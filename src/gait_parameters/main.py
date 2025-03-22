import os
import glob
import json
import pandas as pd
import logging
import warnings

from my_utils.helpers import save_csv
from gait_pipeline import GaitPipeline  # your existing gait pipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=FutureWarning)


def load_config(config_path):
    with open(config_path, 'r') as f:
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
    # Rename gait_parameters folder to "Kinematic_Features"
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
    # EXCLUDE any file name containing "_cropped" so they won't be processed again
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

    # Process each file
    for input_file in input_files:
        summary_df, skipped_file = process_single_file(input_file, config["gait_parameters"]["save_path"], config, analysis_module)
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


def process_single_file(input_file, output_dir, config, module):
    """
    Process a single file based on the selected module.
    For video files, crop if necessary, then run the chosen pipeline.
    
    Returns a tuple: (summary_df, skipped_file)
      - summary_df is the DataFrame with computed parameters (or None if skipped).
      - skipped_file is the new filename if the file was skipped, or None otherwise.
    """
    skipped_file = None

    # Step 1: YOLO-based cropping (common for both modules)
    if input_file.lower().endswith((".mp4", ".mov")):
        from modules.yolo_cropper import YOLOCropper
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
            config['pose_estimator']['image_dimensions'] = cropped_size

    # Step 2: Branch by module selection
    if module == "gait":
        # Gait analysis
        pipeline = GaitPipeline(
            input_path=input_file,
            config=config,
            save_parameters_path=None
        )
        pose_data = pipeline.load_input()
        if pose_data is None:
            logger.info("Skipping %s due to loading issues.", input_file)
            return None, input_file

        pipeline.preprocess()
        pipeline.detect_events()

        # (Optional) Visualization integration if needed...
        if config.get("visualize", False):
            from my_utils.plotting import butter_lowpass_filter, detect_extremas, plot_combined_extremas_and_toe
            from my_utils.prompt_visualisation import prompt_visualisation
            import matplotlib.pyplot as plt

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

        gait_parameters = pipeline.compute_gait_parameters()
        fog_events = pipeline.detect_freezes()
        if fog_events:
            fog_count = len(fog_events)
            fog_total_duration = sum(event['duration_sec'] for event in fog_events)
        else:
            fog_count = 0
            fog_total_duration = 0.0

        median_summary = gait_parameters.median(numeric_only=True)
        if isinstance(median_summary.index, pd.MultiIndex):
            median_summary.index = ['_'.join(map(str, tup)).strip() for tup in median_summary.index.values]
        summary_df = pd.DataFrame(median_summary).T

        video_name = os.path.splitext(os.path.basename(input_file))[0]
        summary_df['video_name'] = video_name
        summary_df['fog_count'] = fog_count
        summary_df['fog_total_duration_sec'] = fog_total_duration

        # Reorder columns to ensure video_name is first
        columns = ['video_name'] + [col for col in summary_df.columns if col != 'video_name']
        summary_df = summary_df[columns]
        logger.info("Processed %s; gait analysis complete.", input_file)
        return summary_df, None

    elif module == "postural_tremor":
        # Run tremor integration for postural tremor analysis.
        from modules.tremor_integration import run_tremor_analysis
        tremor_features = run_tremor_analysis(input_file, config)
        if tremor_features is None:
            logger.error("Tremor analysis failed for %s", input_file)
            return None, input_file

        video_name = os.path.splitext(os.path.basename(input_file))[0]
        if isinstance(tremor_features, pd.DataFrame):
            tremor_features["video_name"] = video_name
            summary_df = tremor_features.reset_index(drop=True)
            if 'video_name' in summary_df.columns:
                columns = ['video_name'] + [col for col in summary_df.columns if col != 'video_name']
                summary_df = summary_df[columns]
            logger.info("Processed %s; postural tremor analysis complete.", input_file)
            return summary_df, None
        else:
            summary_df = pd.DataFrame({
                'video_name': [video_name],
                'dominant_tremor_frequency': [tremor_features.get('dominant_freq', None)],
                'tremor_amplitude': [tremor_features.get('tremor_amplitude', None)],
                'frame_rate': [tremor_features.get('frame_rate', None)],
                'n_frames': [tremor_features.get('n_frames', None)]
            })
            logger.info("Processed %s; postural tremor analysis complete (dictionary fallback).", input_file)
            return summary_df, None

    else:
        logger.error("Invalid module specified: %s", module)
        return None, input_file


if __name__ == "__main__":
    main()
