# src/modules/tremor/feature_extraction.py

import logging
import pandas as pd
import numpy as np
from scipy import signal
import os
import matplotlib.pyplot as plt

from .utils import get_marker_columns, check_hand_
from .signal_analysis import butter_bandpass_filter, pca_tremor_analysis

logger = logging.getLogger(__name__)

def _extract_tremor_features_for_segment(
    p, segment_data: pd.DataFrame, hand: str, plot: bool, save_plots: bool,
    plots_dir: str, feature_prefix: str
) -> dict:
    """Helper to extract tremor features for a given data segment."""
    processed_data = segment_data.interpolate().dropna()

    if processed_data.empty or processed_data.shape[0] < 2:
        logger.warning(
            f"Not enough valid data for {p.patient_id} - {hand} {feature_prefix} after cleaning. Skipping feature extraction."
        )
        return None

    data_detrended = signal.detrend(processed_data.values)
    
    filtered = np.zeros_like(data_detrended)
    for i in range(data_detrended.shape[1]):
        filtered[:, i] = butter_bandpass_filter(data_detrended[:, i], 3, 12, p.sampling_frequency, order=5)
    
    # MODIFIED: Unpack the new spectrum data (f, P)
    features, projection, (f, P) = pca_tremor_analysis(filtered, p.sampling_frequency)
    
    if plot and projection is not None:
        # --- PLOT 1: PCA Projection (existing plot) ---
        plt.figure(figsize=(10, 4))
        plt.plot(projection)
        plt.title(f"PCA Projection: {p.patient_id} - {hand} {feature_prefix}")
        plt.xlabel("Samples")
        plt.ylabel("Amplitude (PCA)")
        if save_plots:
            plot_path = os.path.join(plots_dir, f'pca_{feature_prefix}_{p.patient_id}_{hand}.png')
            plt.savefig(plot_path)
        plt.close()
        
        # --- PLOT 2: Power Spectrum (new plot) ---
        plt.figure(figsize=(10, 4))
        plt.plot(f, P, color='b')
        dominant_freq = features['pca_power_spectral_dominant_frequency']
        plt.axvline(x=dominant_freq, color='r', linestyle='--', label=f'Dominant Freq: {dominant_freq:.2f} Hz')
        plt.title(f"Power Spectrum: {p.patient_id} - {hand} {feature_prefix}")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power")
        plt.xlim(0, 15) # Focus on the relevant tremor frequency range
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        if save_plots:
            plot_path = os.path.join(plots_dir, f'spectrum_{feature_prefix}_{p.patient_id}_{hand}.png')
            plt.savefig(plot_path)
        plt.close()
    
    return features


def extract_proximal_arm_tremor_features(pc, plot=False, save_plots=False, plots_dir=".") -> pd.DataFrame:
    """Extracts proximal arm tremor features (shoulder-elbow)."""
    features_df = pd.DataFrame(index=pc.get_patient_ids())
    for p in pc.patients:
        if not hasattr(p, 'proximal_tremor_features') or p.proximal_tremor_features.empty:
            continue
        
        hands = check_hand_(p)
        for hand in hands:
            markers = [f'{hand}_shoulder', f'{hand}_elbow']
            cols = [col for m in markers for col in get_marker_columns(p.pose_estimation, m)]
            
            if len(cols) >= 2:
                segment_data = p.pose_estimation[cols]
                features = _extract_tremor_features_for_segment(
                    p, segment_data, hand, plot, save_plots, plots_dir, "proximal_arm"
                )
                if features:
                    for key, value in features.items():
                        features_df.loc[p.patient_id, f"{key}_proximal_arm_{hand}"] = value
    return features_df

def extract_distal_arm_tremor_features(pc, plot=False, save_plots=False, plots_dir=".") -> pd.DataFrame:
    """Extracts distal arm tremor features (elbow-wrist)."""
    features_df = pd.DataFrame(index=pc.get_patient_ids())
    for p in pc.patients:
        if not hasattr(p, 'distal_tremor_features') or p.distal_tremor_features.empty:
            continue

        hands = check_hand_(p)
        for hand in hands:
            markers = [f'{hand}_elbow', f'{hand}_wrist']
            cols = [col for m in markers for col in get_marker_columns(p.pose_estimation, m)]
            
            if len(cols) >= 2:
                segment_data = p.pose_estimation[cols]
                features = _extract_tremor_features_for_segment(
                    p, segment_data, hand, plot, save_plots, plots_dir, "distal_arm"
                )
                if features:
                    for key, value in features.items():
                        features_df.loc[p.patient_id, f"{key}_distal_arm_{hand}"] = value
    return features_df

def extract_fingers_tremor_features(pc, plot=False, save_plots=False, plots_dir=".") -> pd.DataFrame:
    """Extracts fingers tremor features using PCA."""
    features_df = pd.DataFrame(index=pc.get_patient_ids())
    for p in pc.patients:
        if not hasattr(p, 'fingers_tremor_features') or p.fingers_tremor_features.empty:
            continue

        segment_data = p.fingers_tremor_features
        left_cols = [c for c in segment_data.columns if 'left' in c[0]]
        right_cols = [c for c in segment_data.columns if 'right' in c[0]]

        if left_cols:
            features = _extract_tremor_features_for_segment(
                p, segment_data[left_cols], 'left', plot, save_plots, plots_dir, "fingers"
            )
            if features:
                for key, value in features.items():
                    features_df.loc[p.patient_id, f"{key}_fingers_left"] = value
        
        if right_cols:
            features = _extract_tremor_features_for_segment(
                p, segment_data[right_cols], 'right', plot, save_plots, plots_dir, "fingers"
            )
            if features:
                for key, value in features.items():
                    features_df.loc[p.patient_id, f"{key}_fingers_right"] = value
    return features_df