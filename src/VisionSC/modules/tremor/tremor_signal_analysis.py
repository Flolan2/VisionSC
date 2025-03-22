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

def spectrogram(x: np.ndarray, fs: float):
    """
    Compute the spectrogram of a signal.
    """
    n = len(x)
    x = np.pad(x, (int(fs/2), int(fs/2)), mode='symmetric')
    amplitudes = []
    frequencies = []
    for i in range(n):
        y = x[i:i+int(fs)]
        fft_result = np.fft.fft(y)
        frequency = np.fft.fftfreq(y.size, d=1/fs)
        amplitude = 2 * np.abs(fft_result) / len(y)
        amplitudes.append(amplitude)
        frequencies.append(frequency)
    amplitude = np.vstack(amplitudes)
    # Use only non-negative frequencies.
    amplitude = amplitude[:, frequency >= 0]
    frequency = frequency[frequency >= 0]
    return frequency, amplitude

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
    """
    if signal_data.shape[1] not in [2, 3]:
        raise ValueError("Input signal must have 2 or 3 dimensions corresponding to coordinates.")
    principal_component, projection = pca_main_component(signal_data)
    analytic_signal = hilbert(projection)
    inst_amplitude = np.abs(analytic_signal)
    max_hilbert_amp = inst_amplitude.max()
    mean_hilbert_amp = inst_amplitude.mean()
    
    f, P = spectrum(projection, fs)
    f_spec, S = spectrogram(projection, fs)
    mean_freq = f_spec[S.mean(axis=0).argmax()]
    max_freq = f_spec[S.max(axis=0).argmax()]
    
    dom_f_idx = P.argmax()
    dominant_frequency = f[dom_f_idx]
    power_spectral_max_amp = P[dom_f_idx]
    
    features = {
        'pca_hilbert_max_amplitude': 2 * max_hilbert_amp,
        'pca_hilbert_mean_amplitude': 2 * mean_hilbert_amp,
        'pca_spectrogram_mean_frequency': mean_freq,
        'pca_spectrogram_max_frequency': max_freq,
        'pca_power_spectral_dominant_frequency': dominant_frequency,
        'pca_power_spectral_max_amplitude': 2 * power_spectral_max_amp
    }
    return features, projection, principal_component
