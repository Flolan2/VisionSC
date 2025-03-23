#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
poet_signal_preprocessing.py
Created on [Date]
Author: [Your Name]

This module contains signal processing and PCA helper functions used for tremor analysis.
"""

import numpy as np
from scipy import signal
from scipy.signal import hilbert, butter, lfilter
import matplotlib.pyplot as plt
from skimage.restoration import denoise_tv_chambolle

def tv1d_denoise(signal_array: np.ndarray, fs: float) -> np.ndarray:
    """
    Perform 1D total variation denoising.
    """
    weight = fs / 1000.0
    signal_2d = signal_array.reshape(-1, 1)
    denoised = denoise_tv_chambolle(signal_2d, weight=weight)
    return denoised.flatten()

def spectrogram(x: np.ndarray, fs: float, nperseg: int = None, noverlap: int = None, window: str = 'hann'):
    """
    Compute the spectrogram of a signal using scipy.signal.spectrogram.
    
    Parameters:
        x (np.ndarray): Input 1D signal.
        fs (float): Sampling frequency.
        nperseg (int, optional): Length of each segment. Defaults to int(fs).
        noverlap (int, optional): Number of points to overlap between segments.
                                  Defaults to nperseg // 2.
        window (str, optional): Desired window to use. Defaults to 'hann'.
        
    Returns:
        f (np.ndarray): Array of sample frequencies (non-negative).
        S (np.ndarray): 2D array of spectrogram amplitudes with shape (n_time_segments, n_frequencies).
                      (Note: This is the transposed output of scipy.signal.spectrogram,
                      so that each row corresponds to a time segment.)
    """
    # Set default segment length and overlap if not provided.
    if nperseg is None:
        nperseg = int(fs)
    if noverlap is None:
        noverlap = nperseg // 2
    
    # Compute the spectrogram using scipy.signal.spectrogram.
    f, t, Sxx = signal.spectrogram(x, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, mode='magnitude')
    
    # Transpose Sxx so that each row corresponds to one time segment, matching the original design.
    S = Sxx.T
    return f, S

def spectrum(y: np.ndarray, fs: float):
    """
    Compute the spectrum of a signal.
    """
    fft_result = np.fft.fft(y)
    frequency = np.fft.fftfreq(y.size, d=1/fs)
    amplitude = 2 * np.abs(fft_result) / len(y)
    amplitude = amplitude[frequency >= 0]
    frequency = frequency[frequency >= 0]
    return frequency, amplitude

def butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 5):
    """
    Create a Butterworth bandpass filter.
    """
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 5) -> np.ndarray:
    """
    Apply a Butterworth bandpass filter to the data.
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def pca_main_component(data: np.ndarray):
    """
    Perform PCA on a 2D array (n_samples x d) and return the first principal component
    and the projection of the data onto that component.
    """
    data_centered = data - np.mean(data, axis=0)
    U, S, Vt = np.linalg.svd(data_centered, full_matrices=False)
    principal_component = Vt[0]
    projection = data_centered.dot(principal_component)
    return principal_component, projection

def pca_tremor_analysis(signal_data: np.ndarray, fs: float):
    """
    Perform PCA on a time series (n_samples x d) and compute tremor features using the projected (1D) signal.
    Now extracts robust features including median and variance measures, adds a frequency variance metric,
    and calculates instantaneous frequency variability.
    
    Parameters:
        signal_data (np.ndarray): Input tremor signal data (n_samples x 2 or 3).
        fs (float): Sampling frequency.
        
    Returns:
        features (dict): A dictionary containing tremor features.
        projection (np.ndarray): The 1D signal obtained after projecting the data onto the first principal component.
        principal_component (np.ndarray): The first principal component.
    """
    if signal_data.shape[1] not in [2, 3]:
        raise ValueError("Input signal must have 2 or 3 dimensions corresponding to coordinates.")
    
    # Perform PCA to extract the dominant tremor pattern.
    principal_component, projection = pca_main_component(signal_data)
    
    # Apply the Hilbert transform to obtain the analytic signal.
    analytic_signal = hilbert(projection)
    inst_amplitude = np.abs(analytic_signal)
    
    # Amplitude-based metrics.
    max_hilbert_amp = inst_amplitude.max()
    median_hilbert_amp = np.median(inst_amplitude)
    amplitude_variance = inst_amplitude.var()
    
    # Compute instantaneous frequency variability using the Hilbert transform.
    # Unwrap the phase to avoid discontinuities, then compute the derivative of phase.
    phase = np.unwrap(np.angle(analytic_signal))
    inst_freq = np.diff(phase) * fs / (2 * np.pi)
    inst_freq_std = np.std(inst_freq)
    
    # Frequency-based metrics from the FFT spectrum.
    f, P = spectrum(projection, fs)
    dom_f_idx = P.argmax()
    dominant_frequency = f[dom_f_idx]
    power_spectral_max_amp = P[dom_f_idx]
    
    # Compute the median frequency from the power spectrum.
    cumulative_power = np.cumsum(P)
    total_power = cumulative_power[-1]
    median_idx = np.searchsorted(cumulative_power, total_power / 2)
    median_frequency = f[median_idx]
    
    # Compute frequency variance as the weighted variance of frequency with weights from P.
    weighted_mean_frequency = np.sum(f * P) / np.sum(P)
    frequency_variance = np.sum(P * (f - weighted_mean_frequency)**2) / np.sum(P)
    
    # Frequency metrics from the spectrogram (optional: keeping only max frequency here).
    f_spec, S = spectrogram(projection, fs)
    max_freq = f_spec[S.max(axis=0).argmax()]
    
    features = {
        'pca_hilbert_max_amplitude': 2 * max_hilbert_amp,
        'pca_hilbert_median_amplitude': 2 * median_hilbert_amp,
        'pca_hilbert_amplitude_variance': amplitude_variance,
        'pca_instantaneous_frequency_std': inst_freq_std,
        'pca_power_spectral_dominant_frequency': dominant_frequency,
        'pca_power_spectral_median_frequency': median_frequency,
        'pca_power_spectral_frequency_variance': frequency_variance,
        'pca_power_spectral_max_amplitude': 2 * power_spectral_max_amp,
        'pca_spectrogram_max_frequency': max_freq  # Optional: if you wish to retain this metric.
    }
    return features, projection, principal_component
