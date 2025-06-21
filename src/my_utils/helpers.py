# --- START OF MODIFIED helpers.py ---

import os
import pandas as pd
import numpy as np
import json
import shutil
import skvideo
import subprocess
import logging # Import logging

from scipy.signal import find_peaks
from scipy.signal import butter, lfilter, filtfilt

# Get logger for this module
logger = logging.getLogger(__name__)

# --- Set FFmpeg Path ---
def set_ffmpeg_path():
    try:
        # Check if ffmpeg is available in the system PATH
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path:
            logger.info(f"FFmpeg is available. Path: {ffmpeg_path}")
            # Check if skvideo needs the path set explicitly
            try:
                # skvideo.setFFmpegPath requires a directory
                skvideo.setFFmpegPath(os.path.dirname(ffmpeg_path))
                logger.info("Set FFmpeg path for skvideo.")
            except Exception as e:
                 logger.warning(f"Could not set FFmpeg path for skvideo (may not be necessary): {e}")
        else:
            logger.warning("FFmpeg executable not found in system PATH.")
    except Exception as e: # Catch potential errors during shutil.which or skvideo call
        logger.error(f"Error configuring FFmpeg path: {e}")
    return

# --- Get Video Frame Rate from Metadata (for CSV inputs) ---
def get_metadata_path(file_path):
    """Get the metadata file path corresponding to the input CSV file"""
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    # Handle potential different naming conventions for metadata
    possible_metadata_name = f"{base_name}_metadata.json"
    alt_metadata_name = file_path.replace(".csv", "_metadata.json") # Check adjacent first

    directory = os.path.dirname(file_path)
    metadata_path = os.path.join(directory, possible_metadata_name)

    if os.path.exists(alt_metadata_name):
        return alt_metadata_name
    elif os.path.exists(metadata_path):
        return metadata_path
    else:
        # Try matching the specific naming from pose_estimation.py
        if "_MPtracked" in base_name:
             meta_name_specific = base_name.replace("_MPtracked", "_MPtracked_metadata.json")
             meta_path_specific = os.path.join(directory, meta_name_specific)
             if os.path.exists(meta_path_specific):
                  return meta_path_specific
        logger.debug(f"Metadata file not found at expected locations for {file_path}")
        return None # Return None if not found

def get_frame_rate(file_path):
    """Get the frame rate of a video from its corresponding metadata file"""
    metadata_path = get_metadata_path(file_path)
    if not metadata_path:
        logger.warning(f"No metadata file found for {file_path}")
        return None

    try:
        with open(metadata_path, 'r') as file:
            data = json.load(file)
            fps = data.get('fps')
            if fps is not None:
                # Attempt to convert to float first for flexibility
                try:
                    return float(fps)
                except ValueError:
                     logger.error(f"Invalid FPS value '{fps}' in metadata file {metadata_path}")
                     return None
            else:
                 logger.warning(f"'fps' key not found in metadata file {metadata_path}")
                 return None
    except FileNotFoundError:
        logger.error(f"Metadata file specified but not found: {metadata_path}")
        return None
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from metadata file: {metadata_path}")
        return None
    except Exception as e:
        logger.error(f"Error reading metadata file {metadata_path}: {e}")
        return None

# --- Robust FPS Extraction for Video Files ---
def get_fps_ffprobe(video_path):
    """
    Extracts the frame rate using ffprobe via a subprocess call.
    Returns the FPS as a float or None if failed.
    """
    # Ensure ffmpeg path is available for ffprobe
    ffmpeg_dir = None
    try:
        ffmpeg_path_test = shutil.which("ffmpeg")
        if ffmpeg_path_test:
            ffmpeg_dir = os.path.dirname(ffmpeg_path_test)
    except Exception:
        pass # Ignore if finding ffmpeg fails here

    ffprobe_cmd = shutil.which("ffprobe", path=ffmpeg_dir) # Try finding ffprobe
    if not ffprobe_cmd:
         logger.warning("ffprobe command not found.")
         return None

    cmd = [
        ffprobe_cmd, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True, timeout=10) # Add timeout
        fps_str = result.stdout.strip()
        if not fps_str:
            logger.warning(f"ffprobe returned empty stdout for {video_path}")
            return None
        if '/' in fps_str:
            num, denom = fps_str.split('/')
            if float(denom) == 0: return None # Avoid division by zero
            return float(num) / float(denom)
        else:
            return float(fps_str)
    except FileNotFoundError:
        logger.error("ffprobe executable not found during subprocess run.")
        return None
    except subprocess.CalledProcessError as e:
        logger.error(f"ffprobe error for {video_path}: {e.stderr.strip()}")
        return None
    except subprocess.TimeoutExpired:
        logger.error(f"ffprobe command timed out for {video_path}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error running ffprobe for {video_path}: {e}")
        return None

def get_fps_opencv(video_path):
    """
    Extracts the frame rate using OpenCV's VideoCapture.
    Returns the FPS as a float or None if failed.
    """
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning(f"OpenCV could not open video file: {video_path}")
            return None
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if fps is None or fps <= 0:
             logger.warning(f"OpenCV returned invalid FPS ({fps}) for {video_path}")
             return None
        return float(fps)
    except ImportError:
        logger.warning("OpenCV is not installed, cannot use it for FPS extraction.")
        return None
    except Exception as e:
        logger.error(f"Error using OpenCV for FPS extraction on {video_path}: {e}")
        return None

def get_robust_fps(video_path, tolerance=0.1):
    """
    Combines ffprobe and OpenCV methods to robustly extract the FPS of a video.
    If both methods are available and within tolerance, the ffprobe value is used.
    Otherwise, a fallback is applied. If neither works, raises RuntimeError.

    Args:
        video_path (str): Path to the video file.
        tolerance (float): Relative difference tolerance between the two methods.

    Returns:
        float: Robustly determined frames per second.

    Raises:
        RuntimeError: If FPS cannot be determined by either method.
    """
    fps_ffprobe = get_fps_ffprobe(video_path)
    fps_cv2 = get_fps_opencv(video_path)

    if fps_ffprobe and fps_cv2:
        # Check if ffprobe value is valid before using it for comparison
        if fps_ffprobe > 0:
            if abs(fps_ffprobe - fps_cv2) / fps_ffprobe < tolerance:
                logger.debug(f"FPS values converged (ffprobe={fps_ffprobe:.2f}, OpenCV={fps_cv2:.2f}). Using ffprobe.")
                return fps_ffprobe
            else:
                logger.warning(f"Discrepancy in FPS values: ffprobe={fps_ffprobe:.2f}, OpenCV={fps_cv2:.2f}. Using ffprobe as primary.")
                return fps_ffprobe
        else: # ffprobe value invalid, use cv2 if valid
            logger.warning(f"ffprobe returned invalid FPS ({fps_ffprobe}). Trying OpenCV value.")
            if fps_cv2 and fps_cv2 > 0:
                return fps_cv2
            else:
                 raise RuntimeError(f"Unable to extract valid FPS using either ffprobe or OpenCV for {video_path}.")
    elif fps_ffprobe and fps_ffprobe > 0:
        logger.debug(f"Using FPS from ffprobe ({fps_ffprobe:.2f}) as OpenCV failed.")
        return fps_ffprobe
    elif fps_cv2 and fps_cv2 > 0:
        logger.debug(f"Using FPS from OpenCV ({fps_cv2:.2f}) as ffprobe failed.")
        return fps_cv2
    else:
        raise RuntimeError(f"Unable to extract valid FPS using either ffprobe or OpenCV for {video_path}.")


# --- File Handling Utilities ---
def validate_file_exists(file_path):
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}") # Use logger
        raise FileNotFoundError(f"File not found: {file_path}")
    return True

def load_csv(file_path, header=[0,1]):
    """Load a CSV file into a pandas DataFrame."""
    try:
        validate_file_exists(file_path) # Check before trying to load
        df = pd.read_csv(file_path, header=header)
        logger.debug(f"Successfully loaded CSV: {file_path}")
        return df
    except FileNotFoundError:
        return None # Already logged in validate_file_exists
    except Exception as e:
        logger.error(f"Error loading CSV file {file_path}: {e}", exc_info=True)
        return None

def save_csv(data, file_path):
    """Saves the provided data to a CSV file."""
    try:
        directory = os.path.dirname(file_path)
        if directory: # Ensure directory is not empty string for current dir
            os.makedirs(directory, exist_ok=True)

        if isinstance(data, pd.DataFrame):
            df = data
        elif isinstance(data, dict):
            # Handle simple dict (one row) or list of dicts
            df = pd.DataFrame(data) if isinstance(data.get(list(data.keys())[0]), list) else pd.DataFrame([data])
        else:
            logger.error(f"Unsupported data type for save_csv: {type(data)}")
            return

        df.to_csv(file_path, index=False)
        logger.debug(f"Successfully saved data to CSV: {file_path}")
    except Exception as e:
        logger.error(f"Error saving data to CSV {file_path}: {e}", exc_info=True)


# --- Gait Analysis Utilities ---

# *** MODIFIED detect_extremas function ***
def detect_extremas(signal, prominence=0.1, distance=None):
    """
    Find peaks (HS) and valleys (TO) in the signal using prominence and distance.

    Args:
        signal (np.array): The input signal (e.g., filtered foot displacement).
        prominence (float, optional): Required prominence of peaks/valleys.
                                      Adjust based on signal noise and amplitude. Defaults to 0.1.
        distance (int, optional): Minimum horizontal distance (samples) between peaks/valleys.
                                  Adjust based on expected step/stride frequency and sampling rate.
                                  Defaults to None (no distance constraint).

    Returns:
        tuple: (peaks_indices, valleys_indices)
    """
    if signal is None or len(signal) == 0:
        return np.array([]), np.array([])

    try:
        # Find peaks (representing Heel Strike - HS)
        peaks, _ = find_peaks(signal, prominence=prominence, distance=distance)

        # Find valleys (representing Toe Off - TO)
        # Use the same prominence for valleys in the inverted signal
        # You might need a different prominence for valleys depending on the signal characteristics
        valleys, _ = find_peaks(-signal, prominence=prominence, distance=distance)

        logger.debug(f"Detected {len(peaks)} peaks and {len(valleys)} valleys with prominence={prominence}, distance={distance}")
        return peaks, valleys

    except Exception as e:
        logger.error(f"Error during extrema detection: {e}", exc_info=True)
        return np.array([]), np.array([]) # Return empty arrays on error


# --- Filtering Utilities ---
def butter_lowpass_filter(signal_data, cutoff=3.0, fs=30.0, order=4):
    """
    Apply a lowpass Butterworth filter to a signal (np.array or pd.Series/DataFrame).

    Args:
        signal_data (np.array | pd.Series | pd.DataFrame): Input signal data.
        cutoff (float): Cutoff frequency in Hz.
        fs (float): Sampling frequency in Hz.
        order (int): Order of the filter.

    Returns:
        np.array | pd.Series | pd.DataFrame: Filtered signal data (same type as input).
    """
    nyquist = 0.5 * fs
    if cutoff >= nyquist:
        logger.warning(f"Filter cutoff frequency ({cutoff} Hz) is >= Nyquist frequency ({nyquist} Hz). Filtering will be ineffective. Returning original signal.")
        return signal_data
    if fs <= 0:
        logger.error("Sampling frequency (fs) must be positive.")
        raise ValueError("Sampling frequency (fs) must be positive.")

    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)

    # Handle different input types
    if isinstance(signal_data, np.ndarray):
        if signal_data.ndim == 1:
            # Filter 1D numpy array
            if len(signal_data) > (order*3): # Ensure signal is long enough for filtfilt
                 return filtfilt(b, a, signal_data)
            else:
                 logger.warning(f"Signal length ({len(signal_data)}) too short for filter order ({order}). Returning original signal.")
                 return signal_data
        elif signal_data.ndim == 2:
             # Filter 2D numpy array (column-wise)
             filtered_data = np.zeros_like(signal_data)
             for i in range(signal_data.shape[1]):
                  if len(signal_data[:, i]) > (order*3):
                       filtered_data[:, i] = filtfilt(b, a, signal_data[:, i])
                  else:
                       logger.warning(f"Signal length ({len(signal_data[:, i])}) too short for filter order ({order}) in column {i}. Returning original signal for this column.")
                       filtered_data[:, i] = signal_data[:, i]
             return filtered_data
        else:
             logger.error("Filtering only supported for 1D or 2D numpy arrays.")
             raise ValueError("Input array must be 1D or 2D")

    elif isinstance(signal_data, pd.Series):
        if len(signal_data) > (order*3):
             filtered_values = filtfilt(b, a, signal_data.values)
             return pd.Series(filtered_values, index=signal_data.index, name=signal_data.name)
        else:
             logger.warning(f"Signal length ({len(signal_data)}) too short for filter order ({order}). Returning original Series.")
             return signal_data

    elif isinstance(signal_data, pd.DataFrame):
        filtered_df = signal_data.copy()
        numeric_cols = signal_data.select_dtypes(include=np.number).columns

        # Exclude based on keywords (more robust check)
        exclude_keywords = ["visibility", "presence", "likelihood"] # Add more if needed
        cols_to_filter = []
        for col in numeric_cols:
            col_name_str = "_".join(map(str, col)) if isinstance(col, tuple) else str(col)
            if not any(keyword in col_name_str.lower() for keyword in exclude_keywords):
                cols_to_filter.append(col)

        logger.debug(f"Applying filter to columns: {cols_to_filter}")

        for col in cols_to_filter:
             # Check length for each column
             if len(filtered_df[col]) > (order*3):
                  # Handle potential NaNs before filtering
                  col_data = filtered_df[col].interpolate(method='linear', limit_direction='both') # Interpolate NaNs
                  filtered_df[col] = filtfilt(b, a, col_data.values)
             else:
                  logger.warning(f"Signal length ({len(filtered_df[col])}) too short for filter order ({order}) in column {col}. Skipping filtering for this column.")
        return filtered_df
    else:
        logger.error(f"Unsupported data type for filtering: {type(signal_data)}")
        raise TypeError("Input data must be a numpy array, pandas Series, or pandas DataFrame.")

# --- (Keep other functions like butter_bandpass, log, compute_and_save_summary as they were) ---

def butter_bandpass(lowcut, highcut, fs, order=5):
    """Design a Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    # Ensure low < high and both are valid
    if low >= high or low <= 0 or high >= 1:
         raise ValueError("Invalid frequency limits for bandpass filter")
    return butter(order, [low, high], btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Apply a Butterworth bandpass filter to the data."""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data) # Use lfilter for bandpass, filtfilt might distort phase less if needed
    return y

def log(message, level="INFO"):
    """Simple logging function (consider replacing with standard logging)."""
    print(f"[{level}] {message}")

def compute_and_save_summary(gait_df, video_name, output_dir):
    """
    Computes summary statistics and saves them. (Simplified version)
    """
    if gait_df is None or gait_df.empty:
         logger.warning(f"Gait DataFrame is empty for {video_name}. Cannot compute summary.")
         return None

    try:
        # Exclude 'Step' column if it exists before aggregation
        if 'Step' in gait_df.columns:
             summary_stats = gait_df.drop(columns=['Step']).agg(['mean', 'median'])
        else:
             summary_stats = gait_df.agg(['mean', 'median'])

        summary_stats = summary_stats.dropna(axis=1, how='all')
        if summary_stats.empty:
             logger.warning(f"Summary stats are empty after dropping NaN columns for {video_name}.")
             return None

        # Flatten multi-index columns if necessary
        if isinstance(summary_stats.columns, pd.MultiIndex):
            summary_stats.columns = ["_".join(map(str, col)) for col in summary_stats.columns.values]

        summary_stats = summary_stats.reset_index().rename(columns={'index': 'statistic'})
        summary_stats.insert(0, 'video', video_name)

        summary_csv_path = os.path.join(output_dir, f"{video_name}_gait_summary.csv")
        save_csv(summary_stats, summary_csv_path)
        return summary_stats
    except Exception as e:
        logger.error(f"Error computing summary for {video_name}: {e}", exc_info=True)
        return None
    
import os
import pandas as pd
import numpy as np
import json
import shutil
import skvideo
import subprocess
import logging # Import logging

from scipy.signal import find_peaks
from scipy.signal import butter, lfilter, filtfilt

# Get logger for this module
logger = logging.getLogger(__name__)

# --- Set FFmpeg Path ---
def set_ffmpeg_path():
    try:
        # Check if ffmpeg is available in the system PATH
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path:
            logger.info(f"FFmpeg is available. Path: {ffmpeg_path}")
            # Check if skvideo needs the path set explicitly
            try:
                # skvideo.setFFmpegPath requires a directory
                skvideo.setFFmpegPath(os.path.dirname(ffmpeg_path))
                logger.info("Set FFmpeg path for skvideo.")
            except Exception as e:
                 logger.warning(f"Could not set FFmpeg path for skvideo (may not be necessary): {e}")
        else:
            logger.warning("FFmpeg executable not found in system PATH.")
    except Exception as e: # Catch potential errors during shutil.which or skvideo call
        logger.error(f"Error configuring FFmpeg path: {e}")
    return

# --- Get Video Frame Rate from Metadata (for CSV inputs) ---
def get_metadata_path(file_path):
    """Get the metadata file path corresponding to the input CSV file"""
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    # Handle potential different naming conventions for metadata
    possible_metadata_name = f"{base_name}_metadata.json"
    alt_metadata_name = file_path.replace(".csv", "_metadata.json") # Check adjacent first

    directory = os.path.dirname(file_path)
    metadata_path = os.path.join(directory, possible_metadata_name)

    if os.path.exists(alt_metadata_name):
        return alt_metadata_name
    elif os.path.exists(metadata_path):
        return metadata_path
    else:
        # Try matching the specific naming from pose_estimation.py
        if "_MPtracked" in base_name:
             meta_name_specific = base_name.replace("_MPtracked", "_MPtracked_metadata.json")
             meta_path_specific = os.path.join(directory, meta_name_specific)
             if os.path.exists(meta_path_specific):
                  return meta_path_specific
        logger.debug(f"Metadata file not found at expected locations for {file_path}")
        return None # Return None if not found

def get_frame_rate(file_path):
    """Get the frame rate of a video from its corresponding metadata file"""
    metadata_path = get_metadata_path(file_path)
    if not metadata_path:
        logger.warning(f"No metadata file found for {file_path}")
        return None

    try:
        with open(metadata_path, 'r') as file:
            data = json.load(file)
            fps = data.get('fps')
            if fps is not None:
                # Attempt to convert to float first for flexibility
                try:
                    return float(fps)
                except ValueError:
                     logger.error(f"Invalid FPS value '{fps}' in metadata file {metadata_path}")
                     return None
            else:
                 logger.warning(f"'fps' key not found in metadata file {metadata_path}")
                 return None
    except FileNotFoundError:
        logger.error(f"Metadata file specified but not found: {metadata_path}")
        return None
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from metadata file: {metadata_path}")
        return None
    except Exception as e:
        logger.error(f"Error reading metadata file {metadata_path}: {e}")
        return None

# --- Robust FPS Extraction for Video Files ---
def get_fps_ffprobe(video_path):
    """
    Extracts the frame rate using ffprobe via a subprocess call.
    Returns the FPS as a float or None if failed.
    """
    # Ensure ffmpeg path is available for ffprobe
    ffmpeg_dir = None
    try:
        ffmpeg_path_test = shutil.which("ffmpeg")
        if ffmpeg_path_test:
            ffmpeg_dir = os.path.dirname(ffmpeg_path_test)
    except Exception:
        pass # Ignore if finding ffmpeg fails here

    ffprobe_cmd = shutil.which("ffprobe", path=ffmpeg_dir) # Try finding ffprobe
    if not ffprobe_cmd:
         logger.warning("ffprobe command not found.")
         return None

    cmd = [
        ffprobe_cmd, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True, timeout=10) # Add timeout
        fps_str = result.stdout.strip()
        if not fps_str:
            logger.warning(f"ffprobe returned empty stdout for {video_path}")
            return None
        if '/' in fps_str:
            num, denom = fps_str.split('/')
            if float(denom) == 0: return None # Avoid division by zero
            return float(num) / float(denom)
        else:
            return float(fps_str)
    except FileNotFoundError:
        logger.error("ffprobe executable not found during subprocess run.")
        return None
    except subprocess.CalledProcessError as e:
        logger.error(f"ffprobe error for {video_path}: {e.stderr.strip()}")
        return None
    except subprocess.TimeoutExpired:
        logger.error(f"ffprobe command timed out for {video_path}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error running ffprobe for {video_path}: {e}")
        return None

def get_fps_opencv(video_path):
    """
    Extracts the frame rate using OpenCV's VideoCapture.
    Returns the FPS as a float or None if failed.
    """
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning(f"OpenCV could not open video file: {video_path}")
            return None
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if fps is None or fps <= 0:
             logger.warning(f"OpenCV returned invalid FPS ({fps}) for {video_path}")
             return None
        return float(fps)
    except ImportError:
        logger.warning("OpenCV is not installed, cannot use it for FPS extraction.")
        return None
    except Exception as e:
        logger.error(f"Error using OpenCV for FPS extraction on {video_path}: {e}")
        return None

def get_robust_fps(video_path, tolerance=0.1):
    """
    Combines ffprobe and OpenCV methods to robustly extract the FPS of a video.
    If both methods are available and within tolerance, the ffprobe value is used.
    Otherwise, a fallback is applied. If neither works, raises RuntimeError.

    Args:
        video_path (str): Path to the video file.
        tolerance (float): Relative difference tolerance between the two methods.

    Returns:
        float: Robustly determined frames per second.

    Raises:
        RuntimeError: If FPS cannot be determined by either method.
    """
    fps_ffprobe = get_fps_ffprobe(video_path)
    fps_cv2 = get_fps_opencv(video_path)

    if fps_ffprobe and fps_cv2:
        # Check if ffprobe value is valid before using it for comparison
        if fps_ffprobe > 0:
            if abs(fps_ffprobe - fps_cv2) / fps_ffprobe < tolerance:
                logger.debug(f"FPS values converged (ffprobe={fps_ffprobe:.2f}, OpenCV={fps_cv2:.2f}). Using ffprobe.")
                return fps_ffprobe
            else:
                logger.warning(f"Discrepancy in FPS values: ffprobe={fps_ffprobe:.2f}, OpenCV={fps_cv2:.2f}. Using ffprobe as primary.")
                return fps_ffprobe
        else: # ffprobe value invalid, use cv2 if valid
            logger.warning(f"ffprobe returned invalid FPS ({fps_ffprobe}). Trying OpenCV value.")
            if fps_cv2 and fps_cv2 > 0:
                return fps_cv2
            else:
                 raise RuntimeError(f"Unable to extract valid FPS using either ffprobe or OpenCV for {video_path}.")
    elif fps_ffprobe and fps_ffprobe > 0:
        logger.debug(f"Using FPS from ffprobe ({fps_ffprobe:.2f}) as OpenCV failed.")
        return fps_ffprobe
    elif fps_cv2 and fps_cv2 > 0:
        logger.debug(f"Using FPS from OpenCV ({fps_cv2:.2f}) as ffprobe failed.")
        return fps_cv2
    else:
        raise RuntimeError(f"Unable to extract valid FPS using either ffprobe or OpenCV for {video_path}.")


# --- File Handling Utilities ---
def validate_file_exists(file_path):
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}") # Use logger
        raise FileNotFoundError(f"File not found: {file_path}")
    return True

def load_csv(file_path, header=[0,1]):
    """Load a CSV file into a pandas DataFrame."""
    try:
        validate_file_exists(file_path) # Check before trying to load
        df = pd.read_csv(file_path, header=header)
        logger.debug(f"Successfully loaded CSV: {file_path}")
        return df
    except FileNotFoundError:
        return None # Already logged in validate_file_exists
    except Exception as e:
        logger.error(f"Error loading CSV file {file_path}: {e}", exc_info=True)
        return None

def save_csv(data, file_path):
    """Saves the provided data to a CSV file."""
    try:
        directory = os.path.dirname(file_path)
        if directory: # Ensure directory is not empty string for current dir
            os.makedirs(directory, exist_ok=True)

        if isinstance(data, pd.DataFrame):
            df = data
        elif isinstance(data, dict):
            # Handle simple dict (one row) or list of dicts
            df = pd.DataFrame(data) if isinstance(data.get(list(data.keys())[0]), list) else pd.DataFrame([data])
        else:
            logger.error(f"Unsupported data type for save_csv: {type(data)}")
            return

        df.to_csv(file_path, index=False)
        logger.debug(f"Successfully saved data to CSV: {file_path}")
    except Exception as e:
        logger.error(f"Error saving data to CSV {file_path}: {e}", exc_info=True)


# --- Gait Analysis Utilities ---

# *** MODIFIED detect_extremas function ***
def detect_extremas(signal, prominence=0.1, distance=None):
    """
    Find peaks (HS) and valleys (TO) in the signal using prominence and distance.

    Args:
        signal (np.array): The input signal (e.g., filtered foot displacement).
        prominence (float, optional): Required prominence of peaks/valleys.
                                      Adjust based on signal noise and amplitude. Defaults to 0.1.
        distance (int, optional): Minimum horizontal distance (samples) between peaks/valleys.
                                  Adjust based on expected step/stride frequency and sampling rate.
                                  Defaults to None (no distance constraint).

    Returns:
        tuple: (peaks_indices, valleys_indices)
    """
    if signal is None or len(signal) == 0:
        return np.array([]), np.array([])

    try:
        # Find peaks (representing Heel Strike - HS)
        peaks, _ = find_peaks(signal, prominence=prominence, distance=distance)

        # Find valleys (representing Toe Off - TO)
        # Use the same prominence for valleys in the inverted signal
        # You might need a different prominence for valleys depending on the signal characteristics
        valleys, _ = find_peaks(-signal, prominence=prominence, distance=distance)

        logger.debug(f"Detected {len(peaks)} peaks and {len(valleys)} valleys with prominence={prominence}, distance={distance}")
        return peaks, valleys

    except Exception as e:
        logger.error(f"Error during extrema detection: {e}", exc_info=True)
        return np.array([]), np.array([]) # Return empty arrays on error


# --- Filtering Utilities ---
def butter_lowpass_filter(signal_data, cutoff=3.0, fs=30.0, order=4):
    """
    Apply a lowpass Butterworth filter to a signal (np.array or pd.Series/DataFrame).

    Args:
        signal_data (np.array | pd.Series | pd.DataFrame): Input signal data.
        cutoff (float): Cutoff frequency in Hz.
        fs (float): Sampling frequency in Hz.
        order (int): Order of the filter.

    Returns:
        np.array | pd.Series | pd.DataFrame: Filtered signal data (same type as input).
    """
    nyquist = 0.5 * fs
    if cutoff >= nyquist:
        logger.warning(f"Filter cutoff frequency ({cutoff} Hz) is >= Nyquist frequency ({nyquist} Hz). Filtering will be ineffective. Returning original signal.")
        return signal_data
    if fs <= 0:
        logger.error("Sampling frequency (fs) must be positive.")
        raise ValueError("Sampling frequency (fs) must be positive.")

    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)

    # Handle different input types
    if isinstance(signal_data, np.ndarray):
        if signal_data.ndim == 1:
            # Filter 1D numpy array
            if len(signal_data) > (order*3): # Ensure signal is long enough for filtfilt
                 return filtfilt(b, a, signal_data)
            else:
                 logger.warning(f"Signal length ({len(signal_data)}) too short for filter order ({order}). Returning original signal.")
                 return signal_data
        elif signal_data.ndim == 2:
             # Filter 2D numpy array (column-wise)
             filtered_data = np.zeros_like(signal_data)
             for i in range(signal_data.shape[1]):
                  if len(signal_data[:, i]) > (order*3):
                       filtered_data[:, i] = filtfilt(b, a, signal_data[:, i])
                  else:
                       logger.warning(f"Signal length ({len(signal_data[:, i])}) too short for filter order ({order}) in column {i}. Returning original signal for this column.")
                       filtered_data[:, i] = signal_data[:, i]
             return filtered_data
        else:
             logger.error("Filtering only supported for 1D or 2D numpy arrays.")
             raise ValueError("Input array must be 1D or 2D")

    elif isinstance(signal_data, pd.Series):
        if len(signal_data) > (order*3):
             filtered_values = filtfilt(b, a, signal_data.values)
             return pd.Series(filtered_values, index=signal_data.index, name=signal_data.name)
        else:
             logger.warning(f"Signal length ({len(signal_data)}) too short for filter order ({order}). Returning original Series.")
             return signal_data

    elif isinstance(signal_data, pd.DataFrame):
        filtered_df = signal_data.copy()
        numeric_cols = signal_data.select_dtypes(include=np.number).columns

        # Exclude based on keywords (more robust check)
        exclude_keywords = ["visibility", "presence", "likelihood"] # Add more if needed
        cols_to_filter = []
        for col in numeric_cols:
            col_name_str = "_".join(map(str, col)) if isinstance(col, tuple) else str(col)
            if not any(keyword in col_name_str.lower() for keyword in exclude_keywords):
                cols_to_filter.append(col)

        logger.debug(f"Applying filter to columns: {cols_to_filter}")

        for col in cols_to_filter:
             # Check length for each column
             if len(filtered_df[col]) > (order*3):
                  # Handle potential NaNs before filtering
                  col_data = filtered_df[col].interpolate(method='linear', limit_direction='both') # Interpolate NaNs
                  filtered_df[col] = filtfilt(b, a, col_data.values)
             else:
                  logger.warning(f"Signal length ({len(filtered_df[col])}) too short for filter order ({order}) in column {col}. Skipping filtering for this column.")
        return filtered_df
    else:
        logger.error(f"Unsupported data type for filtering: {type(signal_data)}")
        raise TypeError("Input data must be a numpy array, pandas Series, or pandas DataFrame.")

# --- (Keep other functions like butter_bandpass, log, compute_and_save_summary as they were) ---

def butter_bandpass(lowcut, highcut, fs, order=5):
    """Design a Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    # Ensure low < high and both are valid
    if low >= high or low <= 0 or high >= 1:
         raise ValueError("Invalid frequency limits for bandpass filter")
    return butter(order, [low, high], btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Apply a Butterworth bandpass filter to the data."""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data) # Use lfilter for bandpass, filtfilt might distort phase less if needed
    return y

def log(message, level="INFO"):
    """Simple logging function (consider replacing with standard logging)."""
    print(f"[{level}] {message}")

def compute_and_save_summary(gait_df, video_name, output_dir):
    """
    Computes summary statistics and saves them. (Simplified version)
    """
    if gait_df is None or gait_df.empty:
         logger.warning(f"Gait DataFrame is empty for {video_name}. Cannot compute summary.")
         return None

    try:
        # Exclude 'Step' column if it exists before aggregation
        if 'Step' in gait_df.columns:
             summary_stats = gait_df.drop(columns=['Step']).agg(['mean', 'median'])
        else:
             summary_stats = gait_df.agg(['mean', 'median'])

        summary_stats = summary_stats.dropna(axis=1, how='all')
        if summary_stats.empty:
             logger.warning(f"Summary stats are empty after dropping NaN columns for {video_name}.")
             return None

        # Flatten multi-index columns if necessary
        if isinstance(summary_stats.columns, pd.MultiIndex):
            summary_stats.columns = ["_".join(map(str, col)) for col in summary_stats.columns.values]

        summary_stats = summary_stats.reset_index().rename(columns={'index': 'statistic'})
        summary_stats.insert(0, 'video', video_name)

        summary_csv_path = os.path.join(output_dir, f"{video_name}_gait_summary.csv")
        save_csv(summary_stats, summary_csv_path)
        return summary_stats
    except Exception as e:
        logger.error(f"Error computing summary for {video_name}: {e}", exc_info=True)
        return None



# --- END OF MODIFIED helpers.py ---