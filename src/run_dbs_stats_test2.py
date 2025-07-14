# src/run_dbs_stats.py

import os
import glob
import logging
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind
from warnings import simplefilter

# Ignore specific warnings from pandas and scipy to keep output clean
simplefilter(action="ignore", category=FutureWarning)
simplefilter(action="ignore", category=UserWarning)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def create_pooled_paired_plot(pooled_data, output_dir):
    """
    MODIFIED: Creates a publication-quality plot for Proximal, Distal, AND Fingers data.
    """
    metrics = ['Amplitude', 'Frequency', 'Variability']
    fig, axes = plt.subplots(1, 3, figsize=(24, 8), squeeze=False) # Increased figure width slightly
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        metric_data = pooled_data[pooled_data['metric'] == metric]
        
        if metric_data.empty:
            ax.set_title(f'DBS Effect on {metric}')
            ax.text(0.5, 0.5, 'No Data Found', ha='center', va='center', fontsize=15, color='red', transform=ax.transAxes)
            continue
            
        # --- Data Extraction for all three segments ---
        off_data_prox = metric_data[(metric_data['limb'] == 'Proximal') & (metric_data['dbs_state'] == 'OFF')]['value']
        on_data_prox = metric_data[(metric_data['limb'] == 'Proximal') & (metric_data['dbs_state'] == 'ON')]['value']
        off_data_dist = metric_data[(metric_data['limb'] == 'Distal') & (metric_data['dbs_state'] == 'OFF')]['value']
        on_data_dist = metric_data[(metric_data['limb'] == 'Distal') & (metric_data['dbs_state'] == 'ON')]['value']
        off_data_fing = metric_data[(metric_data['limb'] == 'Fingers') & (metric_data['dbs_state'] == 'OFF')]['value']
        on_data_fing = metric_data[(metric_data['limb'] == 'Fingers') & (metric_data['dbs_state'] == 'ON')]['value']

        # --- Plotting Individual Lines ---
        prox_x = [0, 1]
        dist_x = [2.5, 3.5] # Added spacing
        fing_x = [5, 6]   # Added Fingers x-positions

        for patient_id in metric_data['patient_id'].unique():
            p_data = metric_data[metric_data['patient_id'] == patient_id]
            
            # Plot Proximal if paired
            p_off_prox = p_data[(p_data['limb'] == 'Proximal') & (p_data['dbs_state'] == 'OFF')]['value']
            p_on_prox = p_data[(p_data['limb'] == 'Proximal') & (p_data['dbs_state'] == 'ON')]['value']
            if not p_off_prox.empty and not p_on_prox.empty:
                ax.plot(prox_x, [p_off_prox.iloc[0], p_on_prox.iloc[0]], marker='o', color='lightblue', alpha=0.7, markersize=8, linestyle='-')
            
            # Plot Distal if paired
            p_off_dist = p_data[(p_data['limb'] == 'Distal') & (p_data['dbs_state'] == 'OFF')]['value']
            p_on_dist = p_data[(p_data['limb'] == 'Distal') & (p_data['dbs_state'] == 'ON')]['value']
            if not p_off_dist.empty and not p_on_dist.empty:
                ax.plot(dist_x, [p_off_dist.iloc[0], p_on_dist.iloc[0]], marker='o', color='lightcoral', alpha=0.7, markersize=8, linestyle='-')

            # Plot Fingers if paired
            p_off_fing = p_data[(p_data['limb'] == 'Fingers') & (p_data['dbs_state'] == 'OFF')]['value']
            p_on_fing = p_data[(p_data['limb'] == 'Fingers') & (p_data['dbs_state'] == 'ON')]['value']
            if not p_off_fing.empty and not p_on_fing.empty:
                ax.plot(fing_x, [p_off_fing.iloc[0], p_on_fing.iloc[0]], marker='o', color='lightgreen', alpha=0.7, markersize=8, linestyle='-')

        # --- Plotting Mean Lines ---
        ax.plot(prox_x, [off_data_prox.mean(), on_data_prox.mean()], 'o--', color='blue', label='Proximal Mean', lw=2.5, markersize=10)
        ax.plot(dist_x, [off_data_dist.mean(), on_data_dist.mean()], 'o--', color='red', label='Distal Mean', lw=2.5, markersize=10)
        ax.plot(fing_x, [off_data_fing.mean(), on_data_fing.mean()], 'o--', color='purple', label='Fingers Mean', lw=2.5, markersize=10)
        
        # --- Statistical Annotations ---
        p_prox, p_dist, p_fing = np.nan, np.nan, np.nan
        if len(off_data_prox) > 1 and len(on_data_prox) > 1:
            _, p_prox = ttest_ind(off_data_prox, on_data_prox, nan_policy='omit', equal_var=False)
        if len(off_data_dist) > 1 and len(on_data_dist) > 1:
            _, p_dist = ttest_ind(off_data_dist, on_data_dist, nan_policy='omit', equal_var=False)
        if len(off_data_fing) > 1 and len(on_data_fing) > 1:
            _, p_fing = ttest_ind(off_data_fing, on_data_fing, nan_policy='omit', equal_var=False)

        def format_p(p_val):
            if np.isnan(p_val): return "N/A"
            if p_val < 0.001: return "p < 0.001"
            if p_val < 0.05: return f"p = {p_val:.3f}"
            return "ns"

        y_max = metric_data['value'].max()
        if pd.isna(y_max): y_max = 1
        y_pos = y_max * 1.1

        if not np.isnan(p_prox):
            ax.plot(prox_x, [y_pos, y_pos], lw=1.5, color='blue')
            ax.text(np.mean(prox_x), y_pos, f" {format_p(p_prox)} ", ha='center', va='bottom', color='blue', backgroundcolor='white')
        if not np.isnan(p_dist):
            ax.plot(dist_x, [y_pos * 1.05, y_pos * 1.05], lw=1.5, color='red') # slight offset
            ax.text(np.mean(dist_x), y_pos * 1.05, f" {format_p(p_dist)} ", ha='center', va='bottom', color='red', backgroundcolor='white')
        if not np.isnan(p_fing):
            ax.plot(fing_x, [y_pos, y_pos], lw=1.5, color='purple')
            ax.text(np.mean(fing_x), y_pos, f" {format_p(p_fing)} ", ha='center', va='bottom', color='purple', backgroundcolor='white')

        # --- Final Plot Formatting ---
        ax.set_title(f'DBS Effect on {metric}', fontsize=14)
        ax.set_ylabel('Metric Value', fontsize=12)
        ax.set_xticks(prox_x + dist_x + fing_x)
        ax.set_xticklabels(['Proximal\nOFF', 'Proximal\nON', 'Distal\nOFF', 'Distal\nON', 'Fingers\nOFF', 'Fingers\nON'], rotation=30, ha="right")
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        if metric_data['value'].notna().any():
            ax.set_ylim(bottom=0, top=y_max * 1.35)
        else:
            ax.set_ylim(bottom=0, top=1)

    fig.suptitle('DBS Effect on Tremor Characteristics (Pooled Limb Data)', fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plot_path = output_dir / "DBS_Effect_Summary_Pooled.png"
    plt.savefig(plot_path, dpi=200)
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
    
    paired_df = agg_df.unstack(level='dbs_state')
    
    num_patients = len(paired_df)
    if num_patients == 0:
        logger.error("No patient data found after processing. Exiting.")
        return

    logger.info(f"Found data for {num_patients} unique patients.")
    
    melted_data = []
    
    def parse_metric_name(col_name):
        # MODIFIED: Correctly identify 'fingers'
        if "amplitude" in col_name: metric = "Amplitude"
        elif "frequency_std" in col_name: metric = "Variability"
        elif "frequency" in col_name: metric = "Frequency"
        else: metric = "Unknown"
        
        if "proximal" in col_name:
            limb = "Proximal"
        elif "distal" in col_name:
            limb = "Distal"
        elif "fingers" in col_name:
            limb = "Fingers"
        else:
            limb = "Unknown"
        return metric, limb

    if isinstance(paired_df.columns, pd.MultiIndex):
        for col_base in paired_df.columns.get_level_values(0).unique():
            metric, limb = parse_metric_name(col_base)
            if metric == "Unknown" or limb == "Unknown": continue

            temp_df = paired_df.loc[:, pd.IndexSlice[col_base, :]].copy()
            temp_df.columns = temp_df.columns.droplevel(0)
            temp_df.index.name = 'patient_id'
            temp_df = temp_df.reset_index()
            temp_df = temp_df.melt(id_vars='patient_id', var_name='dbs_state', value_name='value')
            temp_df['metric'] = metric
            temp_df['limb'] = limb
            melted_data.append(temp_df)
    else:
        logger.error("Dataframe columns are not MultiIndex as expected. Cannot melt data.")
        return

    if not melted_data:
        logger.error("No data could be melted for plotting. Check column names in feature CSVs.")
        return
        
    pooled_df = pd.concat(melted_data, ignore_index=True)

    print("\n" + "="*60)
    print("      DBS ON vs. OFF - POOLED LIMB STATISTICS")
    print("="*60 + "\n")

    # MODIFIED: Add 'Fingers' to the statistics loop
    for metric in ['Amplitude', 'Frequency', 'Variability']:
        for limb in ['Proximal', 'Distal', 'Fingers']:
            subset = pooled_df[(pooled_df['metric'] == metric) & (pooled_df['limb'] == limb)]
            off_data = subset[subset['dbs_state'] == 'OFF']['value'].dropna()
            on_data = subset[subset['dbs_state'] == 'ON']['value'].dropna()
            
            print(f"--- {limb} {metric} (N_off={len(off_data)}, N_on={len(on_data)}) ---")
            print(f"Mean OFF: {off_data.mean():.4f}  |  Mean ON: {on_data.mean():.4f}")
            
            if len(off_data) < 2 or len(on_data) < 2:
                print("Result: Not enough data for statistical comparison.")
            else:
                t_stat, p_value = ttest_ind(off_data, on_data, equal_var=False)
                if p_value < 0.05:
                    change_dir = "decreased" if off_data.mean() > on_data.mean() else "increased"
                    print(f"Result: Statistically SIGNIFICANT (p = {p_value:.4f}). Tremor has {change_dir}.")
                else:
                    print(f"Result: Not statistically significant (p = {p_value:.4f}).")
            print("-" * 25 + "\n")
            
    create_pooled_paired_plot(pooled_df, plots_output_dir)

if __name__ == "__main__":
    run_dbs_statistics()