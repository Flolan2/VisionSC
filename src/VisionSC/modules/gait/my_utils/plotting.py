import math
import os
import cv2
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt
from PIL import Image

def butter_lowpass_filter(data, cutoff=3, fs=30, order=4):
    """
    Apply a Butterworth low-pass filter to smooth the signal.
    
    Parameters:
      - data: The input signal.
      - cutoff: The cutoff frequency.
      - fs: Sampling frequency.
      - order: Order of the filter.
      
    Returns:
      - The filtered signal.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def detect_extremas(signal):
    """
    Find peaks and valleys in the filtered signal.
    
    Uses the mean value as a threshold.
    
    Parameters:
      - signal: 1D numpy array.
      
    Returns:
      - peaks: Indices of peaks.
      - valleys: Indices of valleys.
    """
    threshold = np.mean(signal)
    peaks, _ = find_peaks(signal, height=threshold)
    valleys, _ = find_peaks(-signal, height=-threshold)
    return peaks, valleys 

def display_plot_with_cv2(fig):
    """
    Save the matplotlib figure to an in-memory buffer, load it using OpenCV,
    and display it in a native window.
    """
    if fig is None:
        print("No figure was generated. Skipping visualization.", flush=True)
        return

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    pil_img = Image.open(buf)
    img_array = np.array(pil_img)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Print the command before showing the figure
    print("Press any key in the figure window to continue...", flush=True)
    
    cv2.imshow("Plot", img_array)
    while True:
        key = cv2.waitKey(1)
        if key != -1:
            break
    cv2.destroyAllWindows()
    buf.close()

def plot_raw_pose(landmarks, frame_rate, output_dir="plots", show_plot=False):
    """
    (Deprecated) Plot raw vertical displacement signals for heel and toe landmarks.
    
    Not used in the combined figure.
    """
    heel_left_z = landmarks[("left_heel", "z")]
    heel_right_z = landmarks[("right_heel", "z")]
    toe_left_z = landmarks[("left_foot_index", "z")]
    toe_right_z = landmarks[("right_foot_index", "z")]
    
    time = np.array([t / frame_rate for t in range(len(heel_left_z))])
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    axes[0].plot(time, heel_left_z, label="Left Heel (z coordinate)", color='blue', linewidth=2)
    axes[0].plot(time, heel_right_z, label="Right Heel (z coordinate)", color='orange', linewidth=2)
    axes[0].set_xlabel("Time (s)", fontsize=14)
    axes[0].set_ylabel("Vertical Position (z coordinate)", fontsize=14)
    axes[0].set_title("Heel Vertical Position", fontsize=16)
    axes[0].legend(loc='best', fontsize=12)
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    axes[1].plot(time, toe_left_z, label="Left Toe (z coordinate)", color='blue', linewidth=2)
    axes[1].plot(time, toe_right_z, label="Right Toe (z coordinate)", color='orange', linewidth=2)
    axes[1].set_xlabel("Time (s)", fontsize=14)
    axes[1].set_ylabel("Vertical Position (z coordinate)", fontsize=14)
    axes[1].set_title("Toe Vertical Position", fontsize=16)
    axes[1].legend(loc='best', fontsize=12)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "raw_pose.png"), dpi=300)
    if show_plot:
        display_plot_with_cv2(fig)
    plt.close(fig)

def plot_combined_extremas_and_toe(all_forward_movement, all_extrema_data, frame_rate, input_path, output_dir="plots", show_plot=False):
    """
    Create a combined figure that displays both the extremas (peaks and valleys) 
    of the toe signals and the combined toe movements.
    
    The layout is as follows:
      - Top left: Left Toe Forward Movement with Detected Peaks & Valleys.
      - Top right: Right Toe Forward Movement with Detected Peaks & Valleys.
      - Bottom left: Overlay of Left & Right Toe Displacements Over Time.
      - Bottom right: Phase Plot showing the Relationship Between Left & Right Toe Movements.
    
    If output_dir is provided, the figure is saved; if show_plot is True,
    the figure is displayed interactively using cv2.
    
    Parameters:
      - all_forward_movement: dict with keys "TO_left" and "TO_right" containing displacement signals.
      - all_extrema_data: dict with keys "TO_left" and "TO_right" containing extrema data (peaks, valleys).
      - frame_rate: Frames per second.
      - input_path: Used for naming the output file.
      - output_dir: Directory to save the plot (or None for no saving).
      - show_plot: If True, display the plot interactively.
      
    Returns:
      The matplotlib figure object.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Top left: Left Toe Forward Movement with Detected Peaks & Valleys
    ax = axes[0, 0]
    time_left = np.arange(len(all_forward_movement["TO_left"])) / frame_rate
    ax.plot(time_left, all_forward_movement["TO_left"], label="Left Toe", color='blue', linewidth=2)
    peaks_left = all_extrema_data["TO_left"]["peaks"]
    valleys_left = all_extrema_data["TO_left"]["valleys"]
    indices_peaks_left = (peaks_left * frame_rate).astype(int)
    indices_valleys_left = (valleys_left * frame_rate).astype(int)
    ax.scatter(peaks_left, all_forward_movement["TO_left"][indices_peaks_left], color='red', s=50, label="Peaks (Maxima)")
    ax.scatter(valleys_left, all_forward_movement["TO_left"][indices_valleys_left], color='green', s=50, label="Valleys (Minima)")
    ax.set_title("Left Toe Forward Movement: Detected Peaks & Valleys", fontsize=14)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Displacement")
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Top right: Right Toe Forward Movement with Detected Peaks & Valleys
    ax = axes[0, 1]
    time_right = np.arange(len(all_forward_movement["TO_right"])) / frame_rate
    ax.plot(time_right, all_forward_movement["TO_right"], label="Right Toe", color='orange', linewidth=2)
    peaks_right = all_extrema_data["TO_right"]["peaks"]
    valleys_right = all_extrema_data["TO_right"]["valleys"]
    indices_peaks_right = (peaks_right * frame_rate).astype(int)
    indices_valleys_right = (valleys_right * frame_rate).astype(int)
    ax.scatter(peaks_right, all_forward_movement["TO_right"][indices_peaks_right], color='red', s=50, label="Peaks (Maxima)")
    ax.scatter(valleys_right, all_forward_movement["TO_right"][indices_valleys_right], color='green', s=50, label="Valleys (Minima)")
    ax.set_title("Right Toe Forward Movement: Detected Peaks & Valleys", fontsize=14)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Displacement")
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Bottom left: Overlay of Left & Right Toe Displacements Over Time
    ax = axes[1, 0]
    time_combined = np.arange(len(all_forward_movement["TO_left"])) / frame_rate
    ax.plot(time_combined, all_forward_movement["TO_left"], label="Left Toe", color='blue', linewidth=2)
    ax.plot(time_combined, all_forward_movement["TO_right"], label="Right Toe", color='orange', linewidth=2)
    ax.set_title("Overlay of Left & Right Toe Displacements Over Time", fontsize=14)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Displacement")
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Bottom right: Phase Plot showing the Relationship Between Left & Right Toe Movements
    ax = axes[1, 1]
    ax.plot(all_forward_movement["TO_left"], all_forward_movement["TO_right"], color='black', linewidth=2)
    ax.set_title("Phase Plot: Relationship Between Left & Right Toe Movements", fontsize=14)
    ax.set_xlabel("Left Toe Displacement")
    ax.set_ylabel("Right Toe Displacement")
    ax.grid(True, linestyle='--', alpha=0.6)
    
    filename = os.path.basename(input_path).split('.')[0]
    fig.suptitle(f"Combined Extremas and Toe Movements for {filename}", fontsize=16)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, f"{filename}_combined_extremas_toe.png"), dpi=300)
    if show_plot:
        display_plot_with_cv2(fig)
    # Instead of closing the figure, return it.
    return fig

def plot_extrema_frames(extremas_dict, output_dir, frames_dir):
    """
    Plot and save video frames corresponding to the detected extrema.
    
    For each key in extremas_dict, two plots are created (one for peaks and one for valleys)
    and saved in the output directory.
    
    Parameters:
      - extremas_dict: Dict mapping keys to a tuple (peaks, valleys).
      - output_dir: Directory to save the plots.
      - frames_dir: Directory containing the extracted frame images.
    """
    os.makedirs(output_dir, exist_ok=True)

    for key, (peaks, valleys) in extremas_dict.items():
        extrema_types = {'peaks': peaks, 'valleys': valleys}

        for extrema_type, indices in extrema_types.items():
            if len(indices) == 0:
                continue
        
            img_width, img_height = 10, 7
            cols = 4
            rows = math.ceil(len(indices) / cols)
            figsize = (cols * img_width, rows * img_height)

            fig, axes = plt.subplots(rows, cols, figsize=figsize)
            axes = np.array(axes).flatten()

            for i, idx in enumerate(indices):
                frame_filename = os.path.join(frames_dir, f"frame_{int(idx)}.png")
                frame = cv2.imread(frame_filename)
                if frame is not None:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    axes[i].imshow(frame_rgb)
                    axes[i].axis('off')
                    axes[i].set_title(f"Frame {int(idx)}", fontsize=16)
                else:
                    axes[i].axis('off')
            for j in range(i + 1, len(axes)):
                axes[j].axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{key}_{extrema_type}.png"), dpi=300)
            plt.close(fig)

def extract_frames(video_path, output_dir):
    """
    Extract all frames from a video and save them as individual images.
    
    Parameters:
      - video_path: Path to the video file.
      - output_dir: Directory where frames will be saved.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    for idx in range(total_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_filename = os.path.join(output_dir, f"frame_{idx}.png")
            cv2.imwrite(frame_filename, frame)
    
    cap.release()
