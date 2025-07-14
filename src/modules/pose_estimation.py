# src/modules/pose_estimation.py

import numpy as np
import cv2  # MODIFIED: Use OpenCV instead of skvideo
import mediapipe as mp
from typing import Optional, Any, Tuple, List
import subprocess
import threading
import pandas as pd
import sys
import os
import json
import logging
from tqdm import tqdm
import shutil

from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions

from my_utils.mediapipe_landmarks import prepare_empty_dataframe
from my_utils.helpers import get_robust_fps

# --- FFMPEG Path Configuration (no changes needed here) ---
FFMPEG_EXE_PATH = ""
try:
    FFMPEG_EXE_PATH = shutil.which("ffmpeg")
    if FFMPEG_EXE_PATH is None:
        raise FileNotFoundError
except (ImportError, AttributeError, FileNotFoundError):
    logging.warning("shutil.which('ffmpeg') failed. Please ensure FFmpeg is installed and in your system PATH.")
    # As a last resort, you could point to a known location, but relying on PATH is better.
    FFMPEG_EXE_PATH = "ffmpeg" # Fallback to just calling 'ffmpeg'


def get_code_project_root_for_pose_estimator():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def _read_stderr_thread_func(pipe: subprocess.PIPE, capture_list: List[str]):
    try:
        for line in iter(pipe.readline, b''):
            capture_list.append(line.decode(errors='replace'))
    except ValueError: pass
    except Exception: pass
    finally:
        if pipe and not pipe.closed:
            try: pipe.close()
            except Exception: pass


class PoseEstimator:
    def __init__(self, make_video: Optional[bool] = None, make_csv: Optional[bool] = None, plot: bool = False, config: Optional[dict] = None):
        self.config = config or {}
        pose_estimator_config = self.config.get("pose_estimator", {})

        self.make_video = make_video if make_video is not None else pose_estimator_config.get("make_video", True)
        self.make_csv = make_csv if make_csv is not None else pose_estimator_config.get("make_csv", True)
        self.plot = plot

        self.logger = self._setup_logger()

        code_project_root = get_code_project_root_for_pose_estimator()
        external_io_base = os.path.abspath(os.path.join(code_project_root, ".."))
        default_csv_output_dir = os.path.join(external_io_base, "output", "tracked_data", "csv")
        default_video_output_dir = os.path.join(external_io_base, "output", "tracked_data", "video")

        self.tracked_csv_dir = os.path.abspath(pose_estimator_config.get("tracked_csv_dir", default_csv_output_dir))
        self.tracked_video_dir = os.path.abspath(pose_estimator_config.get("tracked_video_dir", default_video_output_dir))

        src_dir = os.path.join(code_project_root, "src")
        self.hand_model_path_template = os.path.abspath(os.path.join(src_dir, "models/hand_landmarker.task"))
        self.pose_model_path_template = os.path.abspath(os.path.join(src_dir, "models/pose_landmarker_heavy.task"))

        self.hands: Optional[mp.tasks.vision.HandLandmarker] = None
        self.pose: Optional[mp.tasks.vision.PoseLandmarker] = None

        self.logger.info(f"PoseEstimator instance created. Models will be initialized per video.")
        self.logger.info(f"  Make CSV: {self.make_csv}, Make Video: {self.make_video}")
        self.logger.info(f"  (Default instance) Output CSV dir: {self.tracked_csv_dir}")
        self.logger.info(f"  (Default instance) Output Video dir: {self.tracked_video_dir}")
        self.logger.info(f"  Hand model template path: {self.hand_model_path_template}")
        self.logger.info(f"  Pose model template path: {self.pose_model_path_template}")

        os.makedirs(self.tracked_csv_dir, exist_ok=True)
        os.makedirs(self.tracked_video_dir, exist_ok=True)

    @staticmethod
    def _setup_logger() -> logging.Logger:
        logger = logging.getLogger("PoseEstimator")
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def _initialize_mp_models_for_video(self):
        self.logger.debug(f"Initializing MediaPipe models for new video stream...")
        try:
            if not os.path.exists(self.pose_model_path_template):
                raise FileNotFoundError(f"Pose model file not found: {self.pose_model_path_template}")
            if not os.path.exists(self.hand_model_path_template):
                raise FileNotFoundError(f"Hand model file not found: {self.hand_model_path_template}")

            if self.pose and hasattr(self.pose, 'close'):
                self.logger.debug("Closing existing PoseLandmarker.")
                self.pose.close()
            if self.hands and hasattr(self.hands, 'close'):
                self.logger.debug("Closing existing HandLandmarker.")
                self.hands.close()

            pose_base_options = mp.tasks.BaseOptions(model_asset_path=self.pose_model_path_template)
            pose_options = mp.tasks.vision.PoseLandmarkerOptions(
                base_options=pose_base_options, running_mode=mp.tasks.vision.RunningMode.VIDEO)
            self.pose = mp.tasks.vision.PoseLandmarker.create_from_options(pose_options)

            hand_base_options = mp.tasks.BaseOptions(model_asset_path=self.hand_model_path_template)
            hand_options = mp.tasks.vision.HandLandmarkerOptions(
                base_options=hand_base_options, num_hands=2, running_mode=mp.tasks.vision.RunningMode.VIDEO)
            self.hands = mp.tasks.vision.HandLandmarker.create_from_options(hand_options)
            self.logger.debug("MediaPipe models initialized/re-initialized.")
        except Exception as e:
            self.logger.error(f"Failed to load/re-initialize MediaPipe models: {e}", exc_info=True)
            raise

    def draw_pose_landmarks_on_image(self, image: np.ndarray, detection_result: Any) -> np.ndarray:
        annotated_image = np.copy(image)
        if not detection_result.pose_landmarks or not detection_result.pose_landmarks[0]:
            return annotated_image
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
            for landmark in detection_result.pose_landmarks[0]
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image, pose_landmarks_proto, solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style()
        )
        return annotated_image

    def draw_hand_landmarks_on_image(self, rgb_image: np.ndarray, detection_result: Any) -> np.ndarray:
        hand_landmarks_list = detection_result.hand_landmarks
        annotated_image = np.copy(rgb_image)
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
                for landmark in hand_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(
                annotated_image, hand_landmarks_proto, solutions.hands.HAND_CONNECTIONS,
                solutions.drawing_styles.get_default_hand_landmarks_style(),
                solutions.drawing_styles.get_default_hand_connections_style()
            )
        return annotated_image

    def process_video(self, video_path: str, tracked_csv_dir: Optional[str] = None, tracked_video_dir: Optional[str] = None) -> Optional[Tuple[Optional[pd.DataFrame], Optional[float]]]:
        self.logger.info(f"--- Starting process_video for: {os.path.basename(video_path)} ---")

        try:
            self._initialize_mp_models_for_video()
        except Exception as model_init_error:
            self.logger.error(f"Cannot proceed with video {os.path.basename(video_path)} due to MediaPipe model init failure: {model_init_error}", exc_info=True)
            return None, None

        current_tracked_csv_dir = os.path.abspath(tracked_csv_dir or self.tracked_csv_dir)
        current_tracked_video_dir = os.path.abspath(tracked_video_dir or self.tracked_video_dir)
        os.makedirs(current_tracked_csv_dir, exist_ok=True)
        os.makedirs(current_tracked_video_dir, exist_ok=True)
        self.logger.info(f"Effective Output CSV dir for this run: {current_tracked_csv_dir}")
        self.logger.info(f"Effective Output Video dir for this run: {current_tracked_video_dir}")

        tracked_csv_path, tracked_video_path = self.prepare_file_paths(video_path, current_tracked_csv_dir, current_tracked_video_dir)

        if self.make_csv and os.path.isfile(tracked_csv_path):
            self.logger.info(f"CSV already exists: {tracked_csv_path}. Loading tracked data.")
            marker_df_local: Optional[pd.DataFrame] = None
            fs_from_meta = 25.0
            try:
                marker_df_local = pd.read_csv(tracked_csv_path, header=[0, 1])
            except Exception:
                try:
                    marker_df_local = pd.read_csv(tracked_csv_path)
                except Exception as e_csv:
                    self.logger.error(f"Failed to load existing CSV {tracked_csv_path}: {e_csv}")
                    return None, None

            metadata_path = tracked_csv_path.replace(".csv", "_metadata.json")
            if os.path.isfile(metadata_path):
                try:
                    with open(metadata_path, "r") as f: metadata = json.load(f)
                    fs_from_meta = float(metadata.get("fps", 25.0))
                    self.logger.info(f"Loaded FPS from existing metadata: {fs_from_meta}")
                except Exception as e_meta:
                     self.logger.warning(f"Could not load FPS from metadata {metadata_path}: {e_meta}. Using default {fs_from_meta}")
            else:
                self.logger.warning(f"Metadata file not found for existing CSV: {metadata_path}. Using default {fs_from_meta} FPS.")
            return marker_df_local, fs_from_meta

        # ========== MODIFIED: VIDEO READING BLOCK ==========
        self.logger.info(f"Reading video file for tracking: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.logger.error(f"Error: Cannot open video file with OpenCV: {video_path}")
            return None, None

        input_frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        input_frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        num_frames_read = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.logger.info(f"Input video properties (via OpenCV): {input_frame_width}x{input_frame_height}, Total Frames: {num_frames_read}")

        frames_list = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # OpenCV reads in BGR format; MediaPipe requires RGB. This is a crucial conversion.
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames_list.append(frame_rgb)
        finally:
            cap.release()
        
        if not frames_list:
            self.logger.error(f"No frames were successfully read from {video_path} using OpenCV. Cannot proceed.")
            return None, None
        
        # Update num_frames_read to the actual number of frames loaded into the list
        num_frames_read = len(frames_list)
        self.logger.info(f"Successfully read {num_frames_read} frames into memory using OpenCV.")
        # ========== END OF MODIFIED BLOCK ==========

        fs = 0.0
        try:
            fs = get_robust_fps(video_path)
            if fs <= 0: raise ValueError("FPS must be positive.")
            self.logger.info(f"Detected FPS for input video (used for raw stream input rate): {fs:.3f} FPS.")
        except Exception as e:
            self.logger.error(f"Could not determine valid FPS for {video_path}: {e}. Cannot proceed.", exc_info=True)
            return None, None

        ffmpeg_process: Optional[subprocess.Popen] = None
        stderr_capture_list: List[str] = []
        stderr_thread: Optional[threading.Thread] = None

        if self.make_video:
            self.logger.info(f"Preparing FFmpeg subprocess for output: {tracked_video_path} (Target Output FPS: {fs:.3f}, Frames: {num_frames_read})")
            ffmpeg_cmd = [
                FFMPEG_EXE_PATH, "-y",
                "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{input_frame_width}x{input_frame_height}",
                "-r", f"{fs:.6f}",
                "-i", "-",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-r", f"{fs:.6f}",
                "-fps_mode", "cfr",
                "-frames:v", str(num_frames_read),
                "-an",
                tracked_video_path
            ]
            self.logger.debug(f"FFmpeg command: {' '.join(ffmpeg_cmd)}")
            try:
                ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                stderr_thread = threading.Thread(target=_read_stderr_thread_func, args=(ffmpeg_process.stderr, stderr_capture_list))
                stderr_thread.daemon = True
                stderr_thread.start()
            except Exception as e:
                self.logger.error(f"Failed to start FFmpeg subprocess: {e}", exc_info=True)
                self.make_video = False

        marker_df, marker_mapping = prepare_empty_dataframe(hands='both', pose=True)
        frames_written_to_pipe = 0

        loop_exception = None
        try:
            for i, image_data in enumerate(tqdm(frames_list, desc=f"Processing {os.path.basename(video_path)}", total=num_frames_read, unit="frame")):
                if image_data is None: continue

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_data)
                frame_ms = int((i / (fs if fs > 0 else 30.0)) * 1000)

                results_hands = self.hands.detect_for_video(mp_image, frame_ms)
                results_pose = self.pose.detect_for_video(mp_image, frame_ms)

                if i % 60 == 0: # Log every 2 seconds (approx)
                   self.logger.debug(f"--- Frame {i} --- Hands detected: {len(results_hands.hand_world_landmarks) if results_hands.hand_world_landmarks else 0}, Pose detected: {'Yes' if results_pose.pose_world_landmarks else 'No'}")

                annotated_image = np.copy(image_data)
                if results_pose.pose_world_landmarks and results_pose.pose_world_landmarks[0]:
                    annotated_image = self.draw_pose_landmarks_on_image(annotated_image, results_pose)
                    for l_idx, landmark in enumerate(results_pose.pose_world_landmarks[0]):
                        marker_name = marker_mapping['pose'].get(l_idx)
                        if marker_name:
                            marker_df.loc[i, (marker_name, 'x')] = landmark.x
                            marker_df.loc[i, (marker_name, 'y')] = landmark.y
                            marker_df.loc[i, (marker_name, 'z')] = landmark.z
                            marker_df.loc[i, (marker_name, 'visibility')] = landmark.visibility
                            marker_df.loc[i, (marker_name, 'presence')] = landmark.presence

                if results_hands.hand_world_landmarks:
                    annotated_image = self.draw_hand_landmarks_on_image(annotated_image, results_hands)
                    for h_idx, hand_landmarks_for_one_hand in enumerate(results_hands.hand_world_landmarks):
                        if h_idx < len(results_hands.handedness):
                            handedness_raw = results_hands.handedness[h_idx][0].category_name
                            handedness_key = f"{handedness_raw.lower()}_hand"
                            if hand_landmarks_for_one_hand:
                                for l_idx, landmark in enumerate(hand_landmarks_for_one_hand):
                                    marker_name = marker_mapping.get(handedness_key, {}).get(l_idx)
                                    if marker_name:
                                        marker_df.loc[i, (marker_name, 'x')] = landmark.x
                                        marker_df.loc[i, (marker_name, 'y')] = landmark.y
                                        marker_df.loc[i, (marker_name, 'z')] = landmark.z
                                        marker_df.loc[i, (marker_name, 'visibility')] = getattr(landmark, 'visibility', np.nan)
                                        marker_df.loc[i, (marker_name, 'presence')] = getattr(landmark, 'presence', np.nan)

                if self.make_video and ffmpeg_process and ffmpeg_process.stdin:
                    if not ffmpeg_process.stdin.closed:
                        # FFmpeg needs BGR, our annotated image is RGB. Convert it back.
                        bgr_image_for_ffmpeg = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
                        ffmpeg_process.stdin.write(bgr_image_for_ffmpeg.tobytes())
                        frames_written_to_pipe += 1
                    else:
                        self.logger.warning("FFmpeg stdin pipe closed by FFmpeg before all frames written.")
                        break
        except BrokenPipeError:
            loop_exception = BrokenPipeError(f"Broken pipe while writing frame {frames_written_to_pipe} to FFmpeg.")
            self.logger.error(str(loop_exception), exc_info=False)
        except ValueError as ve:
            loop_exception = ve
            self.logger.error(f"ValueError in MediaPipe processing loop (frame {frames_written_to_pipe}): {ve}", exc_info=True)
        except Exception as e_loop:
            loop_exception = e_loop
            self.logger.error(f"Unhandled error in processing loop for frame {frames_written_to_pipe}: {e_loop}", exc_info=True)
        finally:
            self.logger.info(f"Frames attempted to write to FFmpeg stdin pipe: {frames_written_to_pipe}")

        if loop_exception is not None and frames_written_to_pipe < num_frames_read:
            self.logger.warning(f"Processing loop exited prematurely due to: {loop_exception}. Video/CSV might be incomplete or not saved.")
            if self.make_video and ffmpeg_process:
                if ffmpeg_process.stdin and not ffmpeg_process.stdin.closed:
                    try: ffmpeg_process.stdin.close()
                    except: pass
                if ffmpeg_process.poll() is None:
                    self.logger.info("Terminating FFmpeg process due to loop error...")
                    ffmpeg_process.terminate()
                    try: ffmpeg_process.wait(timeout=5)
                    except: ffmpeg_process.kill()
                if stderr_thread and stderr_thread.is_alive(): stderr_thread.join(timeout=2)
                final_stderr_str_error_case = "".join(stderr_capture_list).strip()
                if final_stderr_str_error_case: self.logger.info(f"FFmpeg stderr (after loop error):\n{final_stderr_str_error_case}")

            if isinstance(loop_exception, ValueError) and "timestamp" in str(loop_exception).lower():
                self.logger.error("Monotonically increasing timestamp error from MediaPipe. Skipping this video's output.")
                return None, fs

        return_code = -1
        if self.make_video and ffmpeg_process:
            if ffmpeg_process.stdin and not ffmpeg_process.stdin.closed:
                try: ffmpeg_process.stdin.close()
                except Exception as e_close_stdin: self.logger.warning(f"Exception closing FFmpeg stdin: {e_close_stdin}")

            try:
                ffmpeg_process.wait(timeout=120)
                return_code = ffmpeg_process.returncode
            except subprocess.TimeoutExpired:
                self.logger.error(f"FFmpeg process timed out for {tracked_video_path}. Killing process.")
                ffmpeg_process.kill()
                try: ffmpeg_process.wait(timeout=5)
                except: pass
                return_code = ffmpeg_process.returncode if hasattr(ffmpeg_process, 'returncode') and ffmpeg_process.returncode is not None else -1
            except Exception as e_wait:
                self.logger.error(f"Exception during ffmpeg_process.wait(): {e_wait}", exc_info=True)
                return_code = ffmpeg_process.poll() if hasattr(ffmpeg_process, 'poll') and ffmpeg_process.poll() is not None else -1

            if stderr_thread and stderr_thread.is_alive():
                stderr_thread.join(timeout=10)

            final_stderr_str = "".join(stderr_capture_list).strip()
            if final_stderr_str: self.logger.info(f"FFmpeg stderr output:\n{final_stderr_str}")

            if return_code == 0:
                self.logger.info(f"FFmpeg successfully wrote video to {tracked_video_path}")
            else:
                self.logger.error(f"FFmpeg FAILED with return code {return_code} for {tracked_video_path}")

        if self.make_csv:
            try:
                if loop_exception is None and (frames_written_to_pipe == num_frames_read or not self.make_video):
                    df_to_save = marker_df
                elif frames_written_to_pipe > 0:
                    df_to_save = marker_df.iloc[:frames_written_to_pipe]
                else:
                    df_to_save = pd.DataFrame()

                if not df_to_save.empty:
                    df_to_save.to_csv(tracked_csv_path, index=False)
                    self.logger.info(f"Saved pose estimation CSV to {tracked_csv_path} ({len(df_to_save)} rows)")
                    output_video_status = "Not created"
                    if self.make_video: output_video_status = tracked_video_path if return_code == 0 else f"Failed (code {return_code}) or not created/incomplete"
                    metadata_json = {"fps": fs if fs > 0 else 0.0, "source_video_frames": num_frames_read,
                                     "processed_frames_in_csv": len(df_to_save),
                                     "output_video_file": output_video_status}
                    with open(tracked_csv_path.replace(".csv", "_metadata.json"), "w") as f: json.dump(metadata_json, f, indent=4)
                elif loop_exception is None:
                     self.logger.warning(f"No landmark data populated to save in CSV for {tracked_csv_path}, but loop completed.")
                     metadata_json = {"fps": fs if fs > 0 else 0.0, "source_video_frames": num_frames_read,
                                     "processed_frames_in_csv": 0,
                                     "output_video_file": "No landmarks" if not self.make_video else (tracked_video_path if return_code == 0 else f"Failed (code {return_code})")}
                     with open(tracked_csv_path.replace(".csv", "_metadata.json"), "w") as f: json.dump(metadata_json, f, indent=4)
                else:
                    self.logger.warning(f"No data to save in CSV for {tracked_csv_path} due to processing error.")
            except Exception as e_csv_save:
                 self.logger.error(f"Error saving CSV or metadata: {e_csv_save}", exc_info=True)

        self.logger.info(f"--- Finished process_video for: {os.path.basename(video_path)} ---")
        final_fs = fs if fs > 0 else None

        if loop_exception is not None and isinstance(loop_exception, ValueError) and "timestamp" in str(loop_exception).lower():
            return None, final_fs
        elif marker_df is not None and not marker_df.empty:
             if marker_df.notna().any().any():
                return marker_df, final_fs
        return None, final_fs


    def prepare_file_paths(self, video_path: str, csv_dir: str, video_dir: str) -> Tuple[str, str]:
        file_name_no_ext = os.path.splitext(os.path.basename(video_path))[0]
        tracked_suffix = "_MPtracked"
        tracked_csv_path = os.path.join(csv_dir, f"{file_name_no_ext}{tracked_suffix}.csv")
        tracked_video_path = os.path.join(video_dir, f"{file_name_no_ext}{tracked_suffix}.mp4")
        return tracked_csv_path, tracked_video_path

    def batch_video_processing(self, input_directory: str) -> None:
        self.logger.info(f"Starting batch processing in directory: {input_directory}")
        import glob
        video_files_list = []
        for ext in ("*.mp4", "*.mov", "*.MP4", "*.MOV"):
            video_files_list.extend(glob.glob(os.path.join(input_directory, "**", ext), recursive=True))

        if not video_files_list:
            self.logger.warning(f"No video files found in {input_directory}.")
            return
        self.logger.info(f"Found {len(video_files_list)} video files to process.")
        for video_file_path in video_files_list:
            self.logger.info(f"Batch processing: {video_file_path}")
            self.process_video(video_file_path)