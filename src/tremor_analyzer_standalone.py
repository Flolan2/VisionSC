# src/tremor_analyzer_standalone.py

import os
import sys
import glob
import json
import logging
import pandas as pd
import numpy as np
import pathlib

# --- Path Setup ---
try:
    project_root = pathlib.Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
except NameError:
    project_root = pathlib.Path.cwd()
PROJECT_ROOT = project_root

from src.modules.tremor.patients import Patient, PatientCollection
from src.modules.tremor.marker_extraction import extract_proximal_tremor, extract_distal_tremor, extract_fingers_tremor
from src.modules.tremor.feature_extraction import extract_proximal_arm_tremor_features, extract_distal_arm_tremor_features, extract_fingers_tremor_features
from src.modules.tremor.plotting import plot_sweep_overview

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
logger = logging.getLogger(__name__)

def setup_file_logger(log_file_path):
    file_handler = logging.FileHandler(log_file_path, mode='w')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    debug_logger = logging.getLogger('DebugLogger')
    debug_logger.setLevel(logging.DEBUG)
    if debug_logger.hasHandlers():
        debug_logger.handlers.clear()
    debug_logger.addHandler(file_handler)
    return debug_logger

def save_output(df, base_name, output_dir, file_suffix):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{base_name}{file_suffix}")
    df.to_csv(output_path, index=True)
    logger.info(f"Saved tremor CSV to: {output_path}")

# =======================================================
# NEW HELPER FUNCTION
def calculate_arm_retention(df):
    """Calculates data retention specifically for key arm markers."""
    arm_markers = [
        'left_shoulder', 'left_elbow', 'left_wrist',
        'right_shoulder', 'right_elbow', 'right_wrist'
    ]
    # Select only the coordinate columns for these specific markers
    arm_cols = [col for col in df.columns if col[0] in arm_markers and col[1] in ['x', 'y', 'z']]
    
    if not arm_cols:
        return 0.0
        
    arm_df = df[arm_cols]
    retention = (arm_df.notna().sum().sum() / arm_df.size) * 100
    return retention
# =======================================================

def run_tremor_analysis(sweep_cutoff, generate_plots):
    external_io_base_dir = PROJECT_ROOT.parent
    tracked_csv_input_dir = external_io_base_dir / "output" / "tracked_data" / "csv"
    tremor_results_dir = external_io_base_dir / "output" / "tremor_results"
    
    os.makedirs(tremor_results_dir / "csv", exist_ok=True)
    os.makedirs(tremor_results_dir / "plots", exist_ok=True)
    os.makedirs(tremor_results_dir / "logs", exist_ok=True)

    log_file = tremor_results_dir / "logs" / "sweep_analysis_log.txt"
    debug_logger = setup_file_logger(log_file)
    logger.info(f"Detailed sweep analysis will be logged to: {log_file}")
    
    input_csv_files = glob.glob(str(tracked_csv_input_dir / "*_MPtracked.csv"))
    if not input_csv_files:
        logger.warning(f"No tracked CSV files found. Exiting.")
        return

    for csv_file_path in input_csv_files:
        base_name = os.path.basename(csv_file_path).replace("_MPtracked.csv", "")
        logger.info(f"--- Starting Analysis for: {base_name} ---")
        debug_logger.info(f"\n{'='*80}\nSTARTING ANALYSIS FOR: {base_name}\n{'='*80}")

        try:
            pose_data_df_raw = pd.read_csv(csv_file_path, header=[0, 1])
            with open(csv_file_path.replace(".csv", "_metadata.json"), "r") as f:
                fps = json.load(f).get("fps", 30.0)

            if sweep_cutoff:
                logger.info(f"Running in SWEEP mode for {base_name}")
                cutoffs = np.arange(0.1, 1.01, 0.05)
                sweep_results = []

                confidence_metric = None
                for metric in ['visibility', 'presence', 'likelihood']:
                    if any(metric in col[1] for col in pose_data_df_raw.columns):
                        confidence_metric = metric
                        break
                debug_logger.info(f"Using confidence metric: {confidence_metric}")

                for cutoff in cutoffs:
                    debug_logger.info(f"\n{'-'*20} Processing Cutoff: {cutoff:.2f} {'-'*20}")
                    
                    df = pose_data_df_raw.copy()
                    
                    if confidence_metric:
                        for col in df.columns:
                            if col[1] in ['x', 'y', 'z']:
                                confidence_col = (col[0], confidence_metric)
                                if confidence_col in df.columns:
                                    low_confidence_mask = df[confidence_col] < cutoff
                                    df.loc[low_confidence_mask, col] = np.nan
                    
                    # =======================================================
                    # MODIFIED: Use the new, more meaningful retention calculation
                    arm_retention_percent = calculate_arm_retention(df)
                    debug_logger.info(f"LOG 1 (Post-NaN): Arm Marker Retention={arm_retention_percent:.2f}%, "
                                      f"Left Wrist count={df[('left_wrist', 'x')].count()}")
                    # =======================================================
                    
                    normalized_df = df.copy()
                    for col in normalized_df.columns:
                        if col[1] in ['x', 'y', 'z']:
                            series = normalized_df[col]
                            mean, std = series.mean(), series.std()
                            if std != 0:
                                normalized_df[col] = (series - mean) / std
                            else:
                                normalized_df[col] = series - mean
                    
                    patient = Patient(
                        pose_estimation=normalized_df,
                        sampling_frequency=fps,
                        patient_id=base_name,
                        clean=False,
                        normalize=False
                    )

                    pc = PatientCollection()
                    pc.add_patient_list([patient])
                    pc = extract_proximal_tremor(pc)
                    pc = extract_distal_tremor(pc)
                    pc = extract_fingers_tremor(pc)
                    
                    proximal_features = extract_proximal_arm_tremor_features(pc, False, False, "")
                    distal_features = extract_distal_arm_tremor_features(pc, False, False, "")
                    fingers_features = extract_fingers_tremor_features(pc, False, False, "")
                    
                    combined_features = pd.concat([proximal_features, distal_features, fingers_features], axis=1)

                    if not combined_features.empty:
                        res = {'cutoff': cutoff}
                        # Use the arm-specific retention for the final report
                        res['data_retention_percent'] = arm_retention_percent
                        
                        for side in ['left', 'right']:
                            res[f'proximal_amp_{side}'] = combined_features.get(f"pca_hilbert_median_amplitude_proximal_arm_{side}", pd.Series([np.nan])).iloc[0]
                            res[f'distal_amp_{side}'] = combined_features.get(f"pca_hilbert_median_amplitude_distal_arm_{side}", pd.Series([np.nan])).iloc[0]
                            res[f'proximal_freq_{side}'] = combined_features.get(f"pca_power_spectral_dominant_frequency_proximal_arm_{side}", pd.Series([np.nan])).iloc[0]
                            res[f'distal_freq_{side}'] = combined_features.get(f"pca_power_spectral_dominant_frequency_distal_arm_{side}", pd.Series([np.nan])).iloc[0]
                        sweep_results.append(res)
                        debug_logger.info(f"Calculated Features for {cutoff:.2f}: {res}")
                
                if sweep_results:
                    patient_plot_dir = tremor_results_dir / "plots" / base_name
                    overview_df = pd.DataFrame(sweep_results).set_index('cutoff')
                    save_output(overview_df, base_name, patient_plot_dir, "_sweep_overview.csv")
                    if generate_plots:
                        plot_sweep_overview(overview_df, base_name, patient_plot_dir)
            else:
                logger.warning("Single run mode not implemented in this debug version.")
        except Exception as e:
            logger.error(f"An error occurred during analysis for {base_name}: {e}", exc_info=True)
            debug_logger.error(f"FATAL ERROR during analysis for {base_name}: {e}", exc_info=True)
            continue

    logger.info("Standalone tremor analysis process completed.")

if __name__ == "__main__":
    run_tremor_analysis(
        sweep_cutoff=True,
        generate_plots=True
    )