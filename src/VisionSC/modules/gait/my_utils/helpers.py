import os
import pandas as pd
import numpy as np
import json
import shutil
import skvideo
import subprocess  # New import for robust FPS extraction

from scipy.signal import find_peaks
from scipy.signal import butter, lfilter, filtfilt

# --- Set FFmpeg Path ---
def set_ffmpeg_path():
    try:
        # Check if ffmpeg is available in the system PATH
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path:
            print(f"FFmpeg is available. Path: {ffmpeg_path}")
            skvideo.setFFmpegPath(os.path.dirname(ffmpeg_path))
    except FileNotFoundError:
        print("FFmpeg is not found on the system.")
    return

# --- Get Video Frame Rate from Metadata (for CSV inputs) ---  
def get_metadata_path(file_path):
    """Get the metadata file path corresponding to the input CSV file

    Args:
        input_file (str): Path to the input CSV file of the tracked pose estimation

    Returns:
        str: Path to the metadata JSON file
    """
    # Get the file name without extension and directory
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # Construct the metadata file name
    metadata_file = f"{base_name}_metadata.json"
    
    # Get the directory of the input file
    directory = os.path.dirname(file_path)
    
    # Combine directory and metadata file name
    return os.path.join(directory, metadata_file)

def get_frame_rate(file_path):
    """Get the frame rate of a video from its corresponding metadata file

    Args:
        file_path (str): .json file with "metadata" in file name

    Returns:
        int: frame rate of video file
    """
    metadata_path = get_metadata_path(file_path)

    try:
        with open(metadata_path, 'r') as file:
            data = json.load(file)
            fps =  data.get('fps')
            return int(fps) if fps is not None else None
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print('Error reading or parsing file:', e)
        return None

# --- Robust FPS Extraction for Video Files ---
def get_fps_ffprobe(video_path):
    """
    Extracts the frame rate using ffprobe via a subprocess call.
    Returns the FPS as a float.
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe error: {result.stderr.strip()}")
    fps_str = result.stdout.strip()
    if '/' in fps_str:
        num, denom = fps_str.split('/')
        return float(num) / float(denom)
    else:
        return float(fps_str)

def get_fps_opencv(video_path):
    """
    Extracts the frame rate using OpenCV's VideoCapture.
    Returns the FPS as a float.
    """
    import cv2  # Import locally since not all scripts may need cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video file: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps

def get_robust_fps(video_path, tolerance=0.1):
    """
    Combines ffprobe and OpenCV methods to robustly extract the FPS of a video.
    If both methods are available and within tolerance, the ffprobe value is used.
    Otherwise, a fallback is applied.
    
    Args:
        video_path (str): Path to the video file.
        tolerance (float): Relative difference tolerance between the two methods.
    
    Returns:
        float: Robustly determined frames per second.
    """
    try:
        fps_ffprobe = get_fps_ffprobe(video_path)
    except Exception as e:
        print(f"ffprobe extraction failed: {e}")
        fps_ffprobe = None
    
    try:
        fps_cv2 = get_fps_opencv(video_path)
    except Exception as e:
        print(f"OpenCV extraction failed: {e}")
        fps_cv2 = None
    
    if fps_ffprobe and fps_cv2:
        # If the two values are similar (within tolerance), use one; otherwise, warn and choose ffprobe.
        if abs(fps_ffprobe - fps_cv2) / fps_ffprobe < tolerance:
            return fps_ffprobe
        else:
            print(f"Discrepancy in FPS values: ffprobe={fps_ffprobe}, OpenCV={fps_cv2}. Using ffprobe.")
            return fps_ffprobe
    elif fps_ffprobe:
        return fps_ffprobe
    elif fps_cv2:
        return fps_cv2
    else:
        raise RuntimeError("Unable to extract FPS using either method.")

# --- File Handling Utilities ---
def validate_file_exists(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return True

def load_csv(file_path, header=[0,1]):
    """Load a CSV file into a pandas DataFrame.

    Args:
        file_path (str): Path to the CSV file.
        header (list, optional): List of column names. Defaults to None.

    Returns:
        pandas.DataFrame: DataFrame containing the CSV data.
    """
    validate_file_exists(file_path)
    try:
        df = pd.read_csv(file_path, header=header)
        return df
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None

def save_csv(data, file_path):
    """
    Saves the provided data to a CSV file.

    Args:
        data (dict or pd.DataFrame): Data to be saved.
        file_path (str): Path to the CSV file where the data will be saved.
    """
    # Ensure the directory for the file exists
    directory = os.path.dirname(file_path)
    os.makedirs(directory, exist_ok=True)
    
    # If data is already a DataFrame, use it directly
    if isinstance(data, pd.DataFrame):
        df = data
    else:
        df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)

# --- Gait Analysis Utilities ---
def detect_extremas(signal):
    """Find peaks and valleys in the filtered signal."""
    threshold = np.mean(signal)
    peaks, _ = find_peaks(signal, height=threshold)
    valleys, _ = find_peaks(-signal, height=-threshold)
    return peaks, valleys

def detect_peaks(signal, threshold=0.5):
    """
    Detects peaks in a signal above a certain threshold.
    :param signal: numpy array, input signal
    :param threshold: float, minimum peak height
    :return: indices of peaks
    """
    peaks, _ = find_peaks(signal, height=threshold)
    return peaks

# --- Filtering Utilities ---
def butter_lowpass_filter(df, columns=None, cutoff=3, fs=30, order=4):
    """Apply a lowpass Butterworth filter to specified columns of a DataFrame."""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)

    # Automatically exclude columns with these keywords in their name
    exclude_keywords = ["marker", "subindex", "visibility", "presence"]
    
    def should_exclude(column):
        """Check if any level of a MultiIndex column contains an exclusion keyword."""
        return any(keyword in str(level) for level in column for keyword in exclude_keywords)

    # Select numeric columns that do NOT contain excluded keywords
    filtered_columns = [
        col for col in df.select_dtypes(include=[np.number]).columns
        if not should_exclude(col)
    ]

    # Apply the filter to selected columns
    filtered_df = df.copy()
    filtered_df[filtered_columns] = filtered_df[filtered_columns].apply(lambda col: filtfilt(b, a, col.values))

    return filtered_df

def butter_bandpass(lowcut, highcut, fs, order=5):
    """Design a Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Apply a Butterworth bandpass filter to the data."""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# --- Logging Utilities ---
def log(message, level="INFO"):
    """
    Simple logging function.
    :param message: str, message to log
    :param level: str, log level (e.g., 'INFO', 'DEBUG', 'WARNING')
    """
    print(f"[{level}] {message}")

def compute_and_save_summary(gait_df, video_name, output_dir):
    """
    Computes summary statistics (mean and median) for each column in the gait DataFrame and saves them as a CSV file.
    Only columns that contain values will be included in the summary.
    The resulting CSV has an explicit 'statistic' column (with values "mean" and "median") and flattened headers.
    The video name is included as the first column in the CSV.
    
    Args:
        gait_df (pd.DataFrame): DataFrame containing per-stride gait parameters.
        video_name (str): Identifier for the video, used in naming the output file.
        output_dir (str): Directory where the summary CSV should be saved.
    
    Returns:
        pd.DataFrame: The summary statistics DataFrame.
    """
    # Compute summary statistics (mean and median for each column)
    summary_stats = gait_df.agg(['mean', 'median'])
    
    # Drop columns that are completely empty (both mean and median are NaN)
    summary_stats = summary_stats.dropna(axis=1, how='all')
    
    # Flatten multi-index columns (e.g., ('left', 'step_length') becomes 'left_step_length')
    summary_stats.columns = [
        "{}_{}".format(col[0], col[1]) if isinstance(col, tuple) else col 
        for col in summary_stats.columns
    ]
    
    # Reset index to turn the row labels ("mean", "median") into a column and rename it "statistic"
    summary_stats = summary_stats.reset_index().rename(columns={'index': 'statistic'})
    
    # Insert video name as the first column
    summary_stats.insert(0, 'video', video_name)
    
    # Build the save path for the individual summary CSV.
    summary_csv_path = os.path.join(output_dir, f"{video_name}_gait_summary.csv")
    
    # Save using the existing helper
    save_csv(summary_stats, summary_csv_path)
    
    # Return the summary DataFrame so it can be aggregated later.
    return summary_stats
