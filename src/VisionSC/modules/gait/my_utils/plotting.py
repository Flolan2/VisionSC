import math
import os
import cv2
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt
from PIL import Image
import logging

logger = logging.getLogger(__name__) # Initialize logger for this module

def butter_lowpass_filter(data, cutoff=3, fs=30, order=4):
    """
    Apply a Butterworth low-pass filter to smooth the signal.

    Parameters:
      - data: The input signal (numpy array).
      - cutoff: The cutoff frequency.
      - fs: Sampling frequency.
      - order: Order of the filter.

    Returns:
      - The filtered signal.
    """
    data = np.asarray(data)
    if data.ndim != 1:
        raise ValueError("Input data must be 1-dimensional.")
    if len(data) <= order * 3:
         logger.warning(f"Data length ({len(data)}) is too short for filter order ({order}). Returning original data.")
         return data

    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    if not (0 < normal_cutoff < 1): # Ensure cutoff is valid
        # If fs is 0 or invalid, normal_cutoff could be inf or nan
        logger.error(f"Invalid normal_cutoff frequency ({normal_cutoff}). Check cutoff ({cutoff}) and fs ({fs}). Returning original data.")
        return data
        # raise ValueError(f"Normalized cutoff frequency ({normal_cutoff}) must be between 0 and 1. Check cutoff ({cutoff}) and fs ({fs}).")

    try:
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return filtfilt(b, a, data)
    except ValueError as e:
         logger.error(f"Error during filtering: {e}. Check input data, cutoff={cutoff}, fs={fs}, order={order}", exc_info=True)
         # Return original data might be safer than raising error in pipeline
         return data

def detect_extremas(signal):
    """
    Find peaks and valleys in the filtered signal using the mean as a threshold.

    Parameters:
      - signal: 1D numpy array.

    Returns:
      - peaks: Indices of peaks.
      - valleys: Indices of valleys.
    """
    signal = np.asarray(signal)
    if signal.ndim != 1 or len(signal) == 0:
        return np.array([]), np.array([])

    if np.all(np.isnan(signal)): # Handle all NaN case
        return np.array([]), np.array([])

    # Handle signals with very low variance (almost flat)
    if np.nanstd(signal) < 1e-6:
         logger.debug("Signal variance is near zero, no extremas detected.")
         return np.array([]), np.array([])

    try:
        # Use nanmean for threshold calculation
        threshold = np.nanmean(signal)
        if pd.isna(threshold): # Handle case where signal might become all NaNs after some operation
             logger.warning("Could not calculate mean threshold (signal might be all NaN). No extremas detected.")
             return np.array([]), np.array([])

        # find_peaks doesn't handle NaNs well, use pandas dropna then find_peaks if needed
        # Or assume signal passed here is clean
        peaks, _ = find_peaks(signal, height=threshold)
        valleys, _ = find_peaks(-signal, height=-threshold) # Threshold for valleys should be based on -signal mean
        return peaks, valleys
    except Exception as e:
        logger.error(f"Error detecting extremas: {e}", exc_info=True)
        return np.array([]), np.array([])


def _save_plot(fig, output_dir, filename, dpi=300):
    """Helper function to save a matplotlib figure."""
    try:
        # Ensure output_dir is treated as the directory path
        full_path = os.path.join(output_dir, filename)
        os.makedirs(os.path.dirname(full_path), exist_ok=True) # Ensure directory exists
        fig.savefig(full_path, dpi=dpi, bbox_inches='tight')
        logger.debug(f"Saved plot: {full_path}")
    except Exception as e:
        logger.error(f"Failed to save plot {filename} to {output_dir}: {e}", exc_info=True)

# --- Plot for Event Signals (formerly validation plot) ---
def plot_combined_extremas_and_toe(all_forward_movement, all_extrema_data, frame_rate, input_path, output_dir="plots", save_plot=True):
    """
    Create a combined figure showing event detection signals (forward toe movement, peaks, valleys).
    Saves the plot automatically if save_plot is True and output_dir is provided.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor('white')

    if "TO_left" not in all_forward_movement or "TO_right" not in all_forward_movement:
        logger.error("Missing 'TO_left' or 'TO_right' in all_forward_movement data for plotting.")
        plt.close(fig)
        return None

    # Top left: Left Toe Forward Movement
    ax = axes[0, 0]
    sig_left = all_forward_movement["TO_left"]
    time_left = np.arange(len(sig_left)) / frame_rate
    ax.plot(time_left, sig_left, label="Left Toe (Filtered)", color='blue', linewidth=1.5)
    if "TO_left" in all_extrema_data and all_extrema_data["TO_left"]:
        peaks_left = all_extrema_data["TO_left"]["peaks"]
        valleys_left = all_extrema_data["TO_left"]["valleys"]
        # Ensure indices are within bounds before accessing signal data
        indices_peaks_left = np.clip((peaks_left * frame_rate).astype(int), 0, len(sig_left) - 1)
        indices_valleys_left = np.clip((valleys_left * frame_rate).astype(int), 0, len(sig_left) - 1)
        if len(indices_peaks_left) > 0:
            ax.scatter(peaks_left, sig_left[indices_peaks_left], color='red', s=40, label="Peaks (HS)", zorder=5)
        if len(indices_valleys_left) > 0:
            ax.scatter(valleys_left, sig_left[indices_valleys_left], color='green', s=40, label="Valleys (TO)", zorder=5)
    ax.set_title("Left Toe Forward Movement & Events", fontsize=12)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Displacement (Rel. Sacrum)")
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.6)

    # Top right: Right Toe Forward Movement
    ax = axes[0, 1]
    sig_right = all_forward_movement["TO_right"]
    time_right = np.arange(len(sig_right)) / frame_rate
    ax.plot(time_right, sig_right, label="Right Toe (Filtered)", color='orange', linewidth=1.5)
    if "TO_right" in all_extrema_data and all_extrema_data["TO_right"]:
        peaks_right = all_extrema_data["TO_right"]["peaks"]
        valleys_right = all_extrema_data["TO_right"]["valleys"]
        indices_peaks_right = np.clip((peaks_right * frame_rate).astype(int), 0, len(sig_right) - 1)
        indices_valleys_right = np.clip((valleys_right * frame_rate).astype(int), 0, len(sig_right) - 1)
        if len(indices_peaks_right) > 0:
            ax.scatter(peaks_right, sig_right[indices_peaks_right], color='red', s=40, label="Peaks (HS)", zorder=5)
        if len(indices_valleys_right) > 0:
             ax.scatter(valleys_right, sig_right[indices_valleys_right], color='green', s=40, label="Valleys (TO)", zorder=5)
    ax.set_title("Right Toe Forward Movement & Events", fontsize=12)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Displacement (Rel. Sacrum)")
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.6)

    # Bottom left: Overlay
    ax = axes[1, 0]
    min_len = min(len(sig_left), len(sig_right))
    time_combined = np.arange(min_len) / frame_rate
    ax.plot(time_combined, sig_left[:min_len], label="Left Toe", color='blue', linewidth=1.5, alpha=0.8)
    ax.plot(time_combined, sig_right[:min_len], label="Right Toe", color='orange', linewidth=1.5, alpha=0.8)
    ax.set_title("Overlay of Toe Displacements", fontsize=12)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Displacement")
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.6)

    # Bottom right: Phase Plot
    ax = axes[1, 1]
    ax.plot(sig_left[:min_len], sig_right[:min_len], color='black', linewidth=1)
    ax.set_title("Phase Plot (Left vs Right Toe)", fontsize=12)
    ax.set_xlabel("Left Toe Displacement")
    ax.set_ylabel("Right Toe Displacement")
    ax.grid(True, linestyle='--', alpha=0.6)
    try: # Set aspect ratio, handle potential errors if data range is zero
         ax.set_aspect('equal', adjustable='box')
    except ValueError:
         logger.warning("Could not set equal aspect ratio for phase plot (data range might be zero).")


    filename_base = os.path.basename(input_path).split('.')[0]
    fig.suptitle(f"Event Detection Signals: {filename_base}", fontsize=14) # Changed title slightly

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_plot and output_dir is not None:
        plot_filename = f"{filename_base}_combined_extremas_toe.png" # Keep original filename convention
        _save_plot(fig, output_dir, plot_filename) # Use helper

    # Return fig so it can be closed by the caller
    return fig


# --- Kinematic Summary Plotting Functions ---

def plot_parameter_timeseries(gait_df, parameters, output_dir, video_name):
    """Plots selected gait parameters against the step/stride number."""
    if gait_df is None or gait_df.empty:
        logger.warning(f"Gait DataFrame is empty or None for {video_name}. Skipping timeseries plot.")
        return

    num_params = len(parameters)
    if num_params == 0:
         logger.warning(f"No parameters specified for timeseries plot for {video_name}.")
         return

    cols = 2
    rows = math.ceil(num_params / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 3.5 * rows), squeeze=False)
    fig.patch.set_facecolor('white')
    axes = axes.flatten()
    plot_count = 0

    for i, param_base_name in enumerate(parameters):
        ax = axes[i]
        has_data = False
        title_param = param_base_name.replace("_", " ").title()

        # Plot left side if exists
        left_col = ('left', param_base_name)
        if left_col in gait_df.columns:
            data_left = gait_df[left_col].dropna()
            if not data_left.empty:
                ax.plot(data_left.index, data_left.values, 'o-', label=f'Left {title_param}', color='blue', markersize=4, alpha=0.7)
                has_data = True

        # Plot right side if exists
        right_col = ('right', param_base_name)
        if right_col in gait_df.columns:
            data_right = gait_df[right_col].dropna()
            if not data_right.empty:
                ax.plot(data_right.index, data_right.values,'s--', label=f'Right {title_param}', color='orange', markersize=4, alpha=0.7)
                has_data = True

        # Plot asymmetry if exists
        asym_col = ('asymmetry', param_base_name)
        if asym_col in gait_df.columns:
             data_asym = gait_df[asym_col].dropna()
             if not data_asym.empty:
                 # Use a secondary y-axis for asymmetry (%) if scales are very different
                 if has_data and (data_left.mean() > 10 * data_asym.mean() or data_right.mean() > 10 * data_asym.mean()): # Heuristic
                      ax2 = ax.twinx()
                      line_asym = ax2.plot(data_asym.index, data_asym.values, '^-', label=f'Asymmetry {title_param} (%)', color='green', markersize=4, alpha=0.7)
                      ax2.set_ylabel("Asymmetry (%)", fontsize=9, color='green')
                      ax2.tick_params(axis='y', labelcolor='green')
                      # Combine legends
                      lines, labels = ax.get_legend_handles_labels()
                      lines2, labels2 = ax2.get_legend_handles_labels()
                      ax.legend(lines + lines2, labels + labels2, fontsize=8, loc='best')
                 else:
                      ax.plot(data_asym.index, data_asym.values, '^-', label=f'Asymmetry {title_param} (%)', color='green', markersize=4, alpha=0.7)
                 has_data = True # Asymmetry counts as data

        elif param_base_name in gait_df.columns: # Handle single-column params
             data_single = gait_df[param_base_name].dropna()
             if not data_single.empty:
                 ax.plot(data_single.index, data_single.values, 'x-', label=f'{title_param}', color='purple', markersize=4, alpha=0.7)
                 has_data = True

        if has_data:
            ax.set_title(f'{title_param} Over Time', fontsize=10)
            ax.set_xlabel("Step/Stride Index", fontsize=9)
            ax.set_ylabel(title_param, fontsize=9)
            if not ('ax2' in locals() and ax2 is not None and ax2.lines): # Avoid double legend if asymmetry used twinx
                ax.legend(fontsize=8, loc='best')
            ax.grid(True, linestyle='--', alpha=0.5)
            plot_count +=1
        else:
             ax.set_title(f'{title_param} (No Data)', fontsize=10)
             ax.text(0.5, 0.5, 'No data available', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
             ax.set_xticks([])
             ax.set_yticks([])
        # Reset ax2 if it exists for the next loop iteration
        if 'ax2' in locals(): del ax2


    # Hide unused subplots
    for j in range(plot_count, len(axes)):
        fig.delaxes(axes[j])

    if plot_count > 0:
        fig.suptitle(f'Gait Parameter Time Series: {video_name}', fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_filename = f"{video_name}_param_timeseries.png"
        _save_plot(fig, output_dir, plot_filename)
    else:
         logger.warning(f"No data found for any specified timeseries parameters for {video_name}.")

    plt.close(fig)


def plot_parameter_distributions(gait_df, parameters, output_dir, video_name):
    """Plots histograms or KDEs for selected gait parameters."""
    if gait_df is None or gait_df.empty:
        logger.warning(f"Gait DataFrame is empty or None for {video_name}. Skipping distribution plot.")
        return

    num_params = len(parameters)
    if num_params == 0:
         logger.warning(f"No parameters specified for distribution plot for {video_name}.")
         return

    cols = 2
    rows = math.ceil(num_params / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 3.5 * rows), squeeze=False)
    fig.patch.set_facecolor('white')
    axes = axes.flatten()
    plot_count = 0

    for i, param_base_name in enumerate(parameters):
        ax = axes[i]
        data_to_plot = []
        labels = []
        colors = []
        title_param = param_base_name.replace("_", " ").title()

        # Gather data for left, right, asymmetry
        left_col = ('left', param_base_name)
        right_col = ('right', param_base_name)
        asym_col = ('asymmetry', param_base_name)

        if left_col in gait_df.columns:
            data_left = gait_df[left_col].dropna()
            if not data_left.empty:
                data_to_plot.append(data_left.values)
                labels.append(f'Left (n={len(data_left)})')
                colors.append('blue')
        if right_col in gait_df.columns:
             data_right = gait_df[right_col].dropna()
             if not data_right.empty:
                 data_to_plot.append(data_right.values)
                 labels.append(f'Right (n={len(data_right)})')
                 colors.append('orange')
        if asym_col in gait_df.columns:
            data_asym = gait_df[asym_col].dropna()
            if not data_asym.empty:
                data_to_plot.append(data_asym.values)
                labels.append(f'Asymmetry (n={len(data_asym)})')
                colors.append('green')
        elif param_base_name in gait_df.columns: # Handle single column param
             data_single = gait_df[param_base_name].dropna()
             if not data_single.empty:
                  data_to_plot.append(data_single.values)
                  labels.append(f'{title_param} (n={len(data_single)})')
                  colors.append('purple')

        if data_to_plot:
            # Plot overlapping histograms
            max_density = 0
            for data, label, color in zip(data_to_plot, labels, colors):
                 weights = np.ones_like(data) / len(data) # Normalize for density-like histogram
                 counts, bins, patches = ax.hist(data, bins='auto', alpha=0.6, label=label, color=color, density=True) # Use density=True
                 max_density = max(max_density, counts.max())

            ax.set_title(f'Distribution of {title_param}', fontsize=10)
            ax.set_xlabel(f'{title_param}', fontsize=9)
            ax.set_ylabel("Density", fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.set_ylim(0, max_density * 1.1) # Adjust y-limit based on density
            plot_count += 1
        else:
            ax.set_title(f'{title_param} (No Data)', fontsize=10)
            ax.text(0.5, 0.5, 'No data available', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])

    # Hide unused subplots
    for j in range(plot_count, len(axes)):
        fig.delaxes(axes[j])

    if plot_count > 0:
        fig.suptitle(f'Gait Parameter Distributions: {video_name}', fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_filename = f"{video_name}_param_distributions.png"
        _save_plot(fig, output_dir, plot_filename)
    else:
         logger.warning(f"No data found for any specified distribution parameters for {video_name}.")

    plt.close(fig)


def plot_left_right_comparison(gait_df, parameters, output_dir, video_name):
    """Creates box plots comparing left and right side parameters."""
    if gait_df is None or gait_df.empty:
        logger.warning(f"Gait DataFrame is empty or None for {video_name}. Skipping L/R comparison plot.")
        return

    plot_params = []
    for param_base_name in parameters:
        if ('left', param_base_name) in gait_df.columns and ('right', param_base_name) in gait_df.columns:
             # Check if there's actually data for both sides
             if gait_df[('left', param_base_name)].notna().any() or gait_df[('right', param_base_name)].notna().any():
                  plot_params.append(param_base_name)
             else:
                  logger.debug(f"Skipping '{param_base_name}' for L/R plot: No valid data found for either side.")
        else:
             logger.debug(f"Skipping '{param_base_name}' for L/R plot: Left or Right column missing.")


    if not plot_params:
         logger.warning(f"No parameters found with valid data for both sides for {video_name}. Skipping L/R comparison.")
         return

    num_params = len(plot_params)
    cols = min(num_params, 3)
    rows = math.ceil(num_params / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)
    fig.patch.set_facecolor('white')
    axes = axes.flatten()

    for i, param_base_name in enumerate(plot_params):
        ax = axes[i]
        title_param = param_base_name.replace("_", " ").title()
        data_left = gait_df[('left', param_base_name)].dropna()
        data_right = gait_df[('right', param_base_name)].dropna()

        data_to_plot = []
        labels = []
        colors = ['lightblue', 'lightcoral']
        used_colors = []

        if not data_left.empty:
            data_to_plot.append(data_left.values)
            labels.append(f'Left (n={len(data_left)})')
            used_colors.append(colors[0])
        if not data_right.empty:
            data_to_plot.append(data_right.values)
            labels.append(f'Right (n={len(data_right)})')
            used_colors.append(colors[1])

        if data_to_plot: # Need at least one side to plot
             bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, showfliers=True, widths=0.6)

             for patch, color in zip(bp['boxes'], used_colors):
                 patch.set_facecolor(color)
             for median in bp['medians']:
                  median.set_color('black')

             ax.set_title(f'Left vs Right: {title_param}', fontsize=11)
             ax.set_ylabel(f'{title_param}', fontsize=9)
             ax.tick_params(axis='x', labelsize=9)
             ax.grid(True, linestyle='--', axis='y', alpha=0.6)
        else:
             # This case should be prevented by the earlier check, but added defensively
             ax.set_title(f'{title_param} (No Data)', fontsize=10)
             ax.text(0.5, 0.5, 'No data available', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
             ax.set_xticks([])
             ax.set_yticks([])

    # Hide unused subplots
    for j in range(num_params, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f'Left/Right Gait Parameter Comparison: {video_name}', fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_filename = f"{video_name}_left_right_comparison.png"
    _save_plot(fig, output_dir, plot_filename)
    plt.close(fig)


def plot_fog_visualization(forward_disp, fog_events, frame_rate, output_dir, video_name):
    """Plots the forward displacement signal with detected FoG events overlaid."""
    if forward_disp is None or len(forward_disp) == 0:
        logger.warning(f"Forward displacement data is missing or empty for {video_name}. Skipping FoG plot.")
        return
    if frame_rate is None or frame_rate <= 0:
        logger.warning(f"Invalid frame rate ({frame_rate}) for {video_name}. Skipping FoG plot.")
        return

    fig, ax = plt.subplots(figsize=(15, 5))
    fig.patch.set_facecolor('white')
    time = np.arange(len(forward_disp)) / frame_rate

    ax.plot(time, forward_disp, label='Forward Displacement (Avg Feet rel. Sacrum)', color='black', linewidth=1)

    if fog_events and isinstance(fog_events, list) and len(fog_events) > 0:
        # Check if the first event looks like a dictionary with expected keys
        if isinstance(fog_events[0], dict) and 'start_time' in fog_events[0] and 'end_time' in fog_events[0]:
             for idx, event in enumerate(fog_events):
                 label = f'Detected FoG ({len(fog_events)} episodes)' if idx == 0 else '_nolegend_'
                 ax.axvspan(event['start_time'], event['end_time'], color='red', alpha=0.3, label=label)
             ax.legend(fontsize=9) # Show legend only if FoG events are plotted
        else:
            logger.warning("FoG events detected, but format is unexpected. Cannot plot FoG overlay.")
            ax.legend(fontsize=9) # Show legend for displacement only
    else:
        logger.info(f"No FoG events detected or provided for {video_name}.")
        ax.legend(fontsize=9) # Show legend for displacement only


    ax.set_title(f'Forward Displacement and Detected FoG Episodes: {video_name}', fontsize=12)
    ax.set_xlabel("Time (s)", fontsize=10)
    ax.set_ylabel("Displacement", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlim(time.min(), time.max()) # Ensure x-axis covers the data range

    plt.tight_layout()
    plot_filename = f"{video_name}_fog_visualization.png"
    _save_plot(fig, output_dir, plot_filename)
    plt.close(fig)