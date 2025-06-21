# src/tremor_analyzer_standalone.py

import os
import sys
import glob
import json
import logging
import argparse
import pandas as pd
import pathlib

# --- Spyder-Proof Dynamic Path Block ---
try:
    script_dir = pathlib.Path(__file__).resolve().parent
    project_root = script_dir.parent
    if str(script_dir) in sys.path:
        sys.path.remove(str(script_dir))
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
except NameError:
    project_root = pathlib.Path.cwd()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
PROJECT_ROOT = project_root
# --- End of Spyder-Proof Block ---

from src.modules.tremor.patients import Patient, PatientCollection
from src.modules.tremor.marker_extraction import (
    extract_proximal_tremor, extract_distal_tremor, extract_fingers_tremor
)
from src.modules.tremor.feature_extraction import (
    extract_proximal_arm_tremor_features,
    extract_distal_arm_tremor_features,
    extract_fingers_tremor_features
)
from src.modules.tremor.utils import executive_summary
# MODIFIED: Import the new plotting functions
from src.modules.tremor.plotting import plot_summary_bars, plot_radar_summary

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
logger = logging.getLogger(__name__)

def save_output(df, base_name, output_dir, file_suffix, file_type='csv'):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{base_name}{file_suffix}")
    try:
        if file_type == 'csv' and isinstance(df, pd.DataFrame):
            df.to_csv(output_path, index=True)
            logger.info(f"Saved tremor CSV to: {output_path}")
        elif file_type == 'txt' and isinstance(df, str):
            with open(output_path, 'w') as f:
                f.write(df)
            logger.info(f"Saved tremor summary to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save output to {output_path}: {e}")

def run_tremor_analysis(args):
    external_io_base_dir = PROJECT_ROOT.parent
    tracked_csv_input_dir = external_io_base_dir / "output" / "tracked_data" / "csv"
    tremor_csv_output_dir = external_io_base_dir / "output" / "tremor_results" / "csv"
    tremor_summary_output_dir = external_io_base_dir / "output" / "tremor_results" / "summary"
    tremor_plots_output_dir = external_io_base_dir / "output" / "tremor_results" / "plots"

    os.makedirs(tremor_csv_output_dir, exist_ok=True)
    os.makedirs(tremor_summary_output_dir, exist_ok=True)
    os.makedirs(tremor_plots_output_dir, exist_ok=True)

    logger.info(f"Input tracked CSVs from: {tracked_csv_input_dir}")
    logger.info(f"Output tremor features to: {tremor_csv_output_dir}")
    
    input_csv_files = glob.glob(str(tracked_csv_input_dir / "*_MPtracked.csv"))

    if not input_csv_files:
        logger.info(f"No tracked CSV files found in '{tracked_csv_input_dir}'.")
        return

    logger.info(f"Found {len(input_csv_files)} tracked CSV files to process.")

    for csv_file_path in input_csv_files:
        base_name = os.path.basename(csv_file_path).replace("_MPtracked.csv", "")
        logger.info(f"--- Starting Tremor Analysis for: {base_name} ---")
        
        final_csv_path = tremor_csv_output_dir / f"{base_name}_tremor_features.csv"
        if os.path.exists(final_csv_path) and not args.overwrite:
            logger.info(f"Output exists, skipping: {base_name}")
            continue

        try:
            pose_data_df = pd.read_csv(csv_file_path, header=[0, 1])
            metadata_path = csv_file_path.replace(".csv", "_metadata.json")
            with open(metadata_path, "r") as f:
                fps = json.load(f).get("fps", 30.0)
            
            patient = Patient(
                pose_estimation=pose_data_df,
                sampling_frequency=fps,
                patient_id=base_name,
                clean=True, normalize=True, scaling_factor=1.0,
                interpolate_pose=True, likelihood_cutoff=0.8
            )
            
            if patient.disabled:
                logger.warning(f"Patient object for {base_name} was disabled. Skipping.")
                continue

            pc = PatientCollection()
            pc.add_patient_list([patient])
            logger.info(f"Loaded and preprocessed data for {base_name} with FPS: {fps}")
            
            pc = extract_proximal_tremor(pc)
            pc = extract_distal_tremor(pc)
            pc = extract_fingers_tremor(pc)
            logger.info("Marker extraction complete.")
            
            proximal_features = extract_proximal_arm_tremor_features(pc, args.plot, args.plot, tremor_plots_output_dir)
            distal_features = extract_distal_arm_tremor_features(pc, args.plot, args.plot, tremor_plots_output_dir)
            fingers_features = extract_fingers_tremor_features(pc, args.plot, args.plot, tremor_plots_output_dir)
            
            combined_features = pd.concat([proximal_features, distal_features, fingers_features], axis=1)
            logger.info("Feature extraction complete.")
            
            # --- MODIFIED: Call the new summary plotting functions ---
            if not combined_features.empty and args.plot:
                plot_summary_bars(combined_features.iloc[0], base_name, tremor_plots_output_dir)
                plot_radar_summary(combined_features.iloc[0], base_name, tremor_plots_output_dir)
                logger.info("Summary plots generated.")

            summary_text = executive_summary(combined_features)
            
            save_output(combined_features, base_name, tremor_csv_output_dir, "_tremor_features.csv", 'csv')
            save_output(summary_text, base_name, tremor_summary_output_dir, "_tremor_summary.txt", 'txt')
            
            logger.info(f"--- Finished Tremor Analysis for: {base_name} ---")

        except Exception as e:
            logger.error(f"An error occurred during tremor analysis for {base_name}: {e}", exc_info=True)
            continue

    logger.info("Standalone tremor analysis process completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone script for tremor analysis from tracked pose data.")
    
    parser.add_argument("--no-plot", action="store_false", dest="plot", help="Add this flag to disable plot generation.")
    parser.add_argument("--overwrite", action="store_true", help="If set, existing results will be overwritten.")
    
    cli_args = parser.parse_args()
    run_tremor_analysis(cli_args)