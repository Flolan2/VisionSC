# src/modules/tremor/plotting.py

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

L = logging.getLogger(__name__)

# MODIFIED: Add a new function for the sweep overview plot
def plot_sweep_overview(sweep_df: pd.DataFrame, patient_id: str, output_dir: str):
    """
    Creates a plot summarizing the effect of likelihood_cutoff on key tremor features.
    """
    try:
        fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
        fig.suptitle(f'Likelihood Cutoff Sweep Analysis for: {patient_id}', fontsize=16)
        
        # --- 1. Amplitude Plot ---
        ax1 = axes[0]
        for side in ['left', 'right']:
            ax1.plot(sweep_df.index, sweep_df[f'proximal_amp_{side}'], marker='o', linestyle='-', label=f'Proximal Amp ({side.capitalize()})')
            ax1.plot(sweep_df.index, sweep_df[f'distal_amp_{side}'], marker='x', linestyle='--', label=f'Distal Amp ({side.capitalize()})')
        ax1.set_ylabel('Median Amplitude (PCA)')
        ax1.set_title('Tremor Amplitude vs. Likelihood Cutoff')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7)

        # --- 2. Frequency Plot ---
        ax2 = axes[1]
        for side in ['left', 'right']:
            ax2.plot(sweep_df.index, sweep_df[f'proximal_freq_{side}'], marker='o', linestyle='-', label=f'Proximal Freq ({side.capitalize()})')
            ax2.plot(sweep_df.index, sweep_df[f'distal_freq_{side}'], marker='x', linestyle='--', label=f'Distal Freq ({side.capitalize()})')
        ax2.set_ylabel('Dominant Frequency (Hz)')
        ax2.set_title('Tremor Frequency vs. Likelihood Cutoff')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.set_ylim(2, 10) # Typical tremor range

        # --- 3. Data Retention Plot ---
        ax3 = axes[2]
        ax3.plot(sweep_df.index, sweep_df['data_retention_percent'], marker='s', color='green', label='Data Retention')
        ax3.set_ylabel('Data Points Retained (%)')
        ax3.set_xlabel('Likelihood Cutoff Threshold')
        ax3.set_title('Data Retention vs. Likelihood Cutoff')
        ax3.legend()
        ax3.grid(True, linestyle='--', alpha=0.7)
        ax3.set_ylim(0, 105)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save the combined plot
        plot_path = os.path.join(output_dir, f'sweep_overview_{patient_id}.png')
        plt.savefig(plot_path)
        plt.close(fig)
        L.info(f"Saved sweep overview plot to {plot_path}")

    except Exception as e:
        L.error(f"Could not generate sweep overview plot for {patient_id}: {e}", exc_info=True)


def plot_summary_bars(features_df: pd.DataFrame, patient_id: str, output_dir: str):
    # ... (no changes in this function) ...
    """
    Creates a grouped bar chart summarizing tremor amplitude and frequency
    for proximal and distal segments, left vs. right.
    """
    try:
        labels = ['Proximal Arm', 'Distal Arm', 'Fingers']
        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # --- Amplitude Plot ---
        left_amps = [
            features_df.get(f"pca_hilbert_median_amplitude_proximal_arm_left", np.nan),
            features_df.get(f"pca_hilbert_median_amplitude_distal_arm_left", np.nan),
            features_df.get(f"pca_hilbert_median_amplitude_fingers_left", np.nan)
        ]
        right_amps = [
            features_df.get(f"pca_hilbert_median_amplitude_proximal_arm_right", np.nan),
            features_df.get(f"pca_hilbert_median_amplitude_distal_arm_right", np.nan),
            features_df.get(f"pca_hilbert_median_amplitude_fingers_right", np.nan)
        ]

        ax1.bar(x - width/2, left_amps, width, label='Left', color='cornflowerblue')
        ax1.bar(x + width/2, right_amps, width, label='Right', color='salmon')
        ax1.set_ylabel('Median Amplitude (PCA)')
        ax1.set_title('Median Tremor Amplitude by Segment')
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels)
        ax1.legend()
        ax1.grid(axis='y', linestyle='--', alpha=0.7)

        # --- Frequency Plot ---
        left_freqs = [
            features_df.get(f"pca_power_spectral_dominant_frequency_proximal_arm_left", np.nan),
            features_df.get(f"pca_power_spectral_dominant_frequency_distal_arm_left", np.nan),
            features_df.get(f"pca_power_spectral_dominant_frequency_fingers_left", np.nan)
        ]
        right_freqs = [
            features_df.get(f"pca_power_spectral_dominant_frequency_proximal_arm_right", np.nan),
            features_df.get(f"pca_power_spectral_dominant_frequency_distal_arm_right", np.nan),
            features_df.get(f"pca_power_spectral_dominant_frequency_fingers_right", np.nan)
        ]

        ax2.bar(x - width/2, left_freqs, width, label='Left', color='cornflowerblue')
        ax2.bar(x + width/2, right_freqs, width, label='Right', color='salmon')
        ax2.set_ylabel('Frequency (Hz)')
        ax2.set_title('Dominant Tremor Frequency by Segment')
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels)
        ax2.legend()
        ax2.grid(axis='y', linestyle='--', alpha=0.7)

        fig.suptitle(f'Tremor Analysis Summary: {patient_id}', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle
        
        plot_path = os.path.join(output_dir, f'summary_bar_{patient_id}.png')
        plt.savefig(plot_path)
        plt.close(fig)
        L.info(f"Saved summary bar chart to {plot_path}")

    except Exception as e:
        L.error(f"Could not generate summary bar chart for {patient_id}: {e}")

def plot_radar_summary(features_df: pd.DataFrame, patient_id: str, output_dir: str):
    # ... (no changes in this function) ...
    """
    Creates a radar chart comparing key tremor features between left and right sides.
    """
    try:
        labels = [
            'Proximal Amp.', 'Distal Amp.', 
            'Proximal Freq. Var.', 'Distal Freq. Var.'
        ]
        num_vars = len(labels)

        # Get data, default to 0 if missing
        left_data = [
            features_df.get(f"pca_hilbert_median_amplitude_proximal_arm_left", 0),
            features_df.get(f"pca_hilbert_median_amplitude_distal_arm_left", 0),
            features_df.get(f"pca_instantaneous_frequency_std_proximal_arm_left", 0),
            features_df.get(f"pca_instantaneous_frequency_std_distal_arm_left", 0)
        ]
        right_data = [
            features_df.get(f"pca_hilbert_median_amplitude_proximal_arm_right", 0),
            features_df.get(f"pca_hilbert_median_amplitude_distal_arm_right", 0),
            features_df.get(f"pca_instantaneous_frequency_std_proximal_arm_right", 0),
            features_df.get(f"pca_instantaneous_frequency_std_distal_arm_right", 0)
        ]

        # Angles for the radar chart
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

        # The plot is a circle, so we need to "complete the loop"
        left_data += left_data[:1]
        right_data += right_data[:1]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        
        # Plot data
        ax.plot(angles, left_data, color='cornflowerblue', linewidth=2, linestyle='solid', label='Left')
        ax.fill(angles, left_data, 'cornflowerblue', alpha=0.25)
        ax.plot(angles, right_data, color='salmon', linewidth=2, linestyle='solid', label='Right')
        ax.fill(angles, right_data, 'salmon', alpha=0.25)
        
        # Formatting
        ax.set_yticklabels([]) # Hide radial ticks
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_title(f'Left vs. Right Tremor Profile: {patient_id}', size=15, y=1.1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

        plot_path = os.path.join(output_dir, f'summary_radar_{patient_id}.png')
        plt.savefig(plot_path)
        plt.close(fig)
        L.info(f"Saved summary radar chart to {plot_path}")

    except Exception as e:
        L.error(f"Could not generate summary radar chart for {patient_id}: {e}")