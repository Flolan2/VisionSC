#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 08:16:34 2025

@author: Lange_L
"""

import numpy as np
import pandas as pd
import logging
from scipy.fft import rfft, rfftfreq
from scipy.signal import butter, filtfilt

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def lowpass_filter(data, cutoff, fs, order=4):
    """
    Apply a Butterworth low-pass filter.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

class FreezingDetector:
    """
    Detects Freezing of Gait (FoG) events from a forward displacement signal.
    
    It uses a sliding window approach to compute:
      - Forward velocity (via the gradient of the displacement).
      - A Freeze Index (FI) via spectral analysis (FFT) of the displacement signal.
    
    A FoG event is flagged when:
      - The mean forward velocity in a window is below `velocity_threshold`.
      - The computed Freeze Index exceeds `fi_threshold`.
    """
    def __init__(self, frame_rate, window_size_sec=2.0, step_size_sec=0.5, 
                 velocity_threshold=0.1, fi_threshold=2.0):
        """
        Parameters:
          - frame_rate: Frames per second of the input signal.
          - window_size_sec: Size of the sliding window in seconds.
          - step_size_sec: Step size for the sliding window in seconds.
          - velocity_threshold: Mean forward velocity (in units/sec) below which a freeze is suspected.
          - fi_threshold: Freeze Index threshold above which a freeze is suspected.
        """
        self.frame_rate = frame_rate
        self.window_size = int(window_size_sec * frame_rate)
        self.step_size = int(step_size_sec * frame_rate)
        self.velocity_threshold = velocity_threshold
        self.fi_threshold = fi_threshold

    def compute_freeze_index(self, signal_window):
        """
        Compute the Freeze Index (FI) for a given window of the signal.
        
        Freeze Index = (Power in freeze band [3-8 Hz]) / (Power in locomotor band [0.5-3 Hz])
        """
        N = len(signal_window)
        yf = rfft(signal_window)
        power_spectrum = np.abs(yf) ** 2
        xf = rfftfreq(N, d=1.0 / self.frame_rate)
        
        # Define frequency bands (in Hz)
        freeze_band = (3, 8)
        locomotor_band = (0.5, 3)
        
        freeze_power = np.sum(power_spectrum[(xf >= freeze_band[0]) & (xf < freeze_band[1])])
        locomotor_power = np.sum(power_spectrum[(xf >= locomotor_band[0]) & (xf < locomotor_band[1])])
        if locomotor_power == 0:
            return np.inf
        return freeze_power / locomotor_power

    def detect_freezes(self, forward_displacement):
        """
        Detect FoG episodes in the forward displacement signal.
        
        Parameters:
          - forward_displacement: 1D numpy array representing the forward displacement signal.
        
        Returns:
          - A list of dictionaries, each describing a freeze episode with:
              start_frame, end_frame, start_time, end_time, duration (sec),
              average Freeze Index, and average forward velocity.
        """
        # Compute forward velocity using gradient (scaled by frame rate to get units/sec)
        velocity = np.gradient(forward_displacement) * self.frame_rate
        # Smooth the velocity signal using a low-pass filter
        velocity = lowpass_filter(velocity, cutoff=2.5, fs=self.frame_rate, order=4)
        
        freeze_events = []
        in_freeze = False
        freeze_start = None
        freeze_event_data = None
        
        # Slide through the signal window-by-window
        for start in range(0, len(forward_displacement) - self.window_size + 1, self.step_size):
            end = start + self.window_size
            window_signal = forward_displacement[start:end]
            window_velocity = velocity[start:end]
            
            mean_velocity = np.mean(np.abs(window_velocity))
            fi = self.compute_freeze_index(window_signal)
              
            # Conditions for suspecting a freeze:
            if mean_velocity < self.velocity_threshold and fi > self.fi_threshold:
                if not in_freeze:
                    in_freeze = True
                    freeze_start = start
                    freeze_event_data = {'fi_values': [], 'velocities': []}
                freeze_event_data['fi_values'].append(fi)
                freeze_event_data['velocities'].append(mean_velocity)
            else:
                if in_freeze:
                    # End current freeze event.
                    freeze_end = start + self.window_size  # approximate end frame
                    event = {
                        'start_frame': freeze_start,
                        'end_frame': freeze_end,
                        'start_time': freeze_start / self.frame_rate,
                        'end_time': freeze_end / self.frame_rate,
                        'duration_sec': (freeze_end - freeze_start) / self.frame_rate,
                        'avg_fi': np.mean(freeze_event_data['fi_values']),
                        'avg_velocity': np.mean(freeze_event_data['velocities'])
                    }
                    freeze_events.append(event)
                    logger.info("Detected freeze: %s", event)
                    in_freeze = False
                    freeze_start = None
                    freeze_event_data = None
        
        # If a freeze is in progress at the end of the signal, close it out.
        if in_freeze:
            freeze_end = len(forward_displacement)
            event = {
                'start_frame': freeze_start,
                'end_frame': freeze_end,
                'start_time': freeze_start / self.frame_rate,
                'end_time': freeze_end / self.frame_rate,
                'duration_sec': (freeze_end - freeze_start) / self.frame_rate,
                'avg_fi': np.mean(freeze_event_data['fi_values']),
                'avg_velocity': np.mean(freeze_event_data['velocities'])
            }
            freeze_events.append(event)
            logger.info("Detected freeze at end of signal: %s", event)
        
        return freeze_events