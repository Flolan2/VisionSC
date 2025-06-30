# src/run_dbs_stats.py

import os
import glob
import logging
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_rel
from warnings import simplefilter

# Ignore specific warnings from pandas and scipy to keep output clean
simplefilter(action="ignore", category=FutureWarning)
simplefilter(action="ignore", category=UserWarning)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def create_pooled_paired_plot(pooled_data, output_dir):
    """
    Creates a publication-quality paired plot using pooled limb data,
    grouping by limb segment on the x-axis.
    """
    metrics = ['Amplitude', 'Frequency', 'Variability']
    fig, axes = plt.subplots(1, 3, figsize=(24, 7), squeeze=False)
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        metric_data = pooled_data[pooled_data['metric'] == metric]
        
        if metric_data.empty:
            ax.set_title(f'DBS Effect on {metric}')
            ax.text(0.5, 0.5, 'No Data Found', ha='center', va='center', fontsize=15, color='red', transform=ax.transAxes)
            continue
            
        # --- Data Extraction for Mean Calculation ---
        off_data_prox = metric_data[(metric_data['limb'] == 'Proximal') & (metric_data['dbs_state'] == 'OFF')]['value']
        on_data_prox = metric_data[(metric_data['limb'] == 'Proximal') & (metric_data['dbs_state'] == 'ON')]['value']
        off_data_dist = metric_data[(metric_data['limb'] == 'Distal') & (metric_data['dbs_state'] == 'OFF')]['value']
        on_data_dist = metric_data[(metric_data['limb'] == 'Distal') & (metric_data['dbs_state'] == 'ON')]['value']

        # --- CORRECTED: Plotting Individual Lines ---
        # Define x-axis positions to create visual groups
        prox_x = [0, 1]
        dist_x = [2, 3]

        for patient_id in metric_data['patient_id'].unique():
            # Filter the dataframe for the current patient's data
            p_data = metric_data[metric_data['patient_id'] == patient_id]

            # Get proximal ON/OFF values for this patient
            p_off_prox_val = p_data[(p_data['limb'] == 'Proximal') & (p_data['dbs_state'] == 'OFF')]['value']
            p_on_prox_val = p_data[(p_data['limb'] == 'Proximal') & (p_data['dbs_state'] == 'ON')]['value']
            
            # Get distal ON/OFF values for this patient
            p_off_dist_val = p_data[(p_data['limb'] == 'Distal') & (p_data['dbs_state'] == 'OFF')]['value']
            p_on_dist_val = p_data[(p_data['limb'] == 'Distal') & (p_data['dbs_state'] == 'ON')]['value']

            # Plot if both ON and OFF data exist for the segment
            if not p_off_prox_val.empty and not p_on_prox_val.empty:
                ax.plot(prox_x, [p_off_prox_val.iloc[0], p_on_prox_val.iloc[0]], marker='o', color='lightblue', alpha=0.8, markersize=8)
            
            if not p_off_dist_val.empty and not p_on_dist_val.empty:
                ax.plot(dist_x, [p_off_dist_val.iloc[0], p_on_dist_val.iloc[0]], marker='o', color='lightcoral', alpha=0.8, markersize=8)

        # --- Plotting Mean Lines ---
        ax.plot(prox_x, [off_data_prox.mean(), on_data_prox.mean()], 'o--', color='blue', label='Proximal Mean', lw=2.5, markersize=10)
        ax.plot(dist_x, [off_data_dist.mean(), on_data_dist.mean()], 'o--', color='red', label='Distal Mean', lw=2.5, markersize=10)
        
        # --- Statistical Annotations ---
        t_prox, p_prox = ttest_rel(off_data_prox, on_data_prox, nan_policy='omit')
        t_dist, p_dist = ttest_rel(off_data_dist, on_data_dist, nan_policy='omit')

        def format_p(p_val):
            if np.isnan(p_val): return "N/A"
            if p_val < 0.001: return "p < 0.001"
            if p_val < 0.05: return f"p = {p_val:.3f}"
            return "ns"

        # Position annotations above the data
        y_max = metric_data['value'].max()
        y_pos = y_max * 1.1

        ax.plot(prox_x, [y_pos, y_pos], lw=1.5, color='blue')
        ax.text(np.mean(prox_x), y_pos, f" {format_p(p_prox)} ", ha='center', va='bottom', color='blue', backgroundcolor='white')
        
        ax.plot(dist_x, [y_pos, y_pos], lw=1.5, color='red')
        ax.text(np.mean(dist_x), y_pos, f" {format_p(p_dist)} ", ha='center', va='bottom', color='red', backgroundcolor='white')

        # --- Final Plot Formatting ---
        ax.set_title(f'DBS Effect on {metric}', fontsize=14)
        ax.set_ylabel('Metric Value', fontsize=12)
        ax.set_xticks(prox_x + dist_x)
        ax.set_xticklabels(['Proximal\nOFF', 'Proximal\nON', 'Distal\nOFF', 'Distal\nON'])
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.set_ylim(bottom=0, top=y_max * 1.25)

    fig.suptitle('DBS Effect on Tremor Characteristics (Pooled Limb Data)', fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plot_path = output_dir / "DBS_Effect_Summary_Pooled.png"
    plt.savefig(plot_path, dpi=200) # Increased DPI for publication quality
    logger.info(f"Pooled data visual summary saved to: {plot_path}")
    plt.close(fig)


def run_dbs_statistics():
    try:
        project_root = pathlib.Path(__file__).resolve().parents[1]
    except NameError:
        project_root = pathlib.Path.cwd()
    
    features_dir = project_root.parent / "output" / "tremor_results" / "csv"
    plots_output_dir = project_root.parent / "output" / "tremor_results" / "plots"
    os.makedirs(plots_output_dir, exist_ok=True)

    logger.info(f"Looking for feature CSVs in: {features_dir}")
    all_csv_files = glob.glob(str(features_dir / "*_tremor_features.csv"))
    if not all_csv_files:
        logger.error("No tremor feature CSV files found. Exiting.")
        return

    all_data_df = pd.concat([pd.read_csv(f).assign(filepath=f) for f in all_csv_files])
    logger.info(f"Loaded data from {len(all_csv_files)} files.")

    def parse_filename(filepath):
        basename = os.path.basename(filepath)
        parts = basename.split('_')
        patient_id = parts[0] + '_' + parts[1]
        dbs_state = "UNKNOWN"
        if "on" in basename.lower(): dbs_state = "ON"
        elif "off" in basename.lower(): dbs_state = "OFF"
        return patient_id, dbs_state

    all_data_df[['patient_id', 'dbs_state']] = all_data_df['filepath'].apply(lambda x: pd.Series(parse_filename(x)))
    all_data_df = all_data_df[all_data_df['dbs_state'] != "UNKNOWN"]
    agg_df = all_data_df.groupby(['patient_id', 'dbs_state']).mean(numeric_only=True)
    pivot_df = agg_df.unstack(level='dbs_state')
    paired_df = pivot_df.dropna()
    
    num_patients = len(paired_df)
    if num_patients < 2:
        logger.error(f"Found only {num_patients} patients with complete ON/OFF data.")
        return
        
    logger.info(f"Found {num_patients} patients with paired ON/OFF data. Running pooled statistics...")
    
    melted_data = []
    
    def parse_metric_name(col_name):
        if "amplitude" in col_name: metric = "Amplitude"
        elif "frequency_std" in col_name: metric = "Variability"
        elif "frequency" in col_name: metric = "Frequency"
        else: metric = "Unknown"
        
        limb = "Proximal" if "proximal" in col_name else "Distal"
        return metric, limb

    for col_base in paired_df.columns.get_level_values(0).unique():
        metric, limb = parse_metric_name(col_base)
        if metric == "Unknown": continue

        temp_df = paired_df[[ (col_base, 'OFF'), (col_base, 'ON') ]].copy()
        temp_df.columns = ['OFF', 'ON']
        temp_df.index.name = 'patient_id'
        temp_df = temp_df.reset_index()
        temp_df = temp_df.melt(id_vars='patient_id', var_name='dbs_state', value_name='value')
        temp_df['metric'] = metric
        temp_df['limb'] = limb
        melted_data.append(temp_df)
    
    pooled_df = pd.concat(melted_data, ignore_index=True)

    print("\n" + "="*60)
    print("      DBS ON vs. OFF - POOLED LIMB STATISTICS")
    print("="*60 + "\n")

    for metric in ['Amplitude', 'Frequency', 'Variability']:
        for limb in ['Proximal', 'Distal']:
            subset = pooled_df[(pooled_df['metric'] == metric) & (pooled_df['limb'] == limb)]
            off_data = subset[subset['dbs_state'] == 'OFF']['value']
            on_data = subset[subset['dbs_state'] == 'ON']['value']
            
            if len(off_data) < 2: continue
            
            t_stat, p_value = ttest_rel(off_data, on_data)
            
            print(f"--- {limb} {metric} (N={len(off_data)}) ---")
            print(f"Mean OFF: {off_data.mean():.4f}  |  Mean ON: {on_data.mean():.4f}")
            if p_value < 0.05:
                change_dir = "decreased" if off_data.mean() > on_data.mean() else "increased"
                print(f"Result: Statistically SIGNIFICANT (p = {p_value:.4f}). Tremor has {change_dir}.")
            else:
                print(f"Result: Not statistically significant (p = {p_value:.4f}).")
            print("-" * 25 + "\n")
            
    create_pooled_paired_plot(pooled_df, plots_output_dir)

if __name__ == "__main__":
    run_dbs_statistics()