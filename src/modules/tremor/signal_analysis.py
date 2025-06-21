# src/modules/tremor/signal_analysis.py

import numpy as np
from scipy import signal
from scipy.signal import hilbert, butter, lfilter

def spectrum(y: np.ndarray, fs: float):
    """Compute the spectrum of a signal."""
    fft_result = np.fft.fft(y)
    frequency = np.fft.fftfreq(y.size, d=1/fs)
    amplitude = 2 * np.abs(fft_result) / len(y)
    amplitude = amplitude[frequency >= 0]
    frequency = frequency[frequency >= 0]
    return frequency, amplitude

def butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 5):
    """Create a Butterworth bandpass filter."""
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 5) -> np.ndarray:
    """Apply a Butterworth bandpass filter to the data."""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def pca_main_component(data: np.ndarray):
    """Perform PCA and return the first principal component and projection."""
    data_centered = data - np.mean(data, axis=0)
    U, S, Vt = np.linalg.svd(data_centered, full_matrices=False)
    principal_component = Vt[0]
    projection = data_centered.dot(principal_component)
    return principal_component, projection

def pca_tremor_analysis(signal_data: np.ndarray, fs: float):
    """Perform PCA and compute tremor features from the projected signal."""
    if signal_data.shape[1] < 2:
        raise ValueError("Input signal must have at least 2 dimensions for PCA.")
    
    principal_component, projection = pca_main_component(signal_data)
    
    analytic_signal = hilbert(projection)
    inst_amplitude = np.abs(analytic_signal)
    
    max_hilbert_amp = inst_amplitude.max()
    median_hilbert_amp = np.median(inst_amplitude)
    amplitude_variance = inst_amplitude.var()
    
    phase = np.unwrap(np.angle(analytic_signal))
    inst_freq = np.diff(phase) * fs / (2 * np.pi)
    inst_freq_std = np.std(inst_freq)
    
    f, P = spectrum(projection, fs)
    dom_f_idx = P.argmax()
    dominant_frequency = f[dom_f_idx]
    power_spectral_max_amp = P[dom_f_idx]
    
    cumulative_power = np.cumsum(P)
    total_power = cumulative_power[-1]
    median_idx = np.searchsorted(cumulative_power, total_power / 2)
    median_frequency = f[median_idx]
    
    weighted_mean_frequency = np.sum(f * P) / np.sum(P)
    frequency_variance = np.sum(P * (f - weighted_mean_frequency)**2) / np.sum(P)
    
    features = {
        'pca_hilbert_max_amplitude': 2 * max_hilbert_amp,
        'pca_hilbert_median_amplitude': 2 * median_hilbert_amp,
        'pca_hilbert_amplitude_variance': amplitude_variance,
        'pca_instantaneous_frequency_std': inst_freq_std,
        'pca_power_spectral_dominant_frequency': dominant_frequency,
        'pca_power_spectral_median_frequency': median_frequency,
        'pca_power_spectral_frequency_variance': frequency_variance,
        'pca_power_spectral_max_amplitude': 2 * power_spectral_max_amp,
    }
    # MODIFIED: Return the spectrum data along with other results
    return features, projection, (f, P)