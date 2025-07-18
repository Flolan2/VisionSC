# src/modules/yolo_cropper.py

# --- Start of Path Fix ---
# This block is to ensure that the script can find the 'my_utils' module
# when run directly from an IDE (like Spyder), especially in a nested directory structure.
import sys
import os
import pathlib
import shutil

try:
    # Resolve the path to the 'src' directory and add it to sys.path
    # This file ('yolo_cropper.py') is in 'src/modules/', so we go up two levels.
    src_path = pathlib.Path(__file__).resolve().parent.parent
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
except NameError:
    # Fallback for environments where __file__ is not defined (e.g., some interactive consoles)
    # This assumes the current working directory is the project root.
    src_path = pathlib.Path.cwd() / 'src'
    if src_path.exists() and str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
# --- End of Path Fix ---


import cv2
import torch
import numpy as np
import logging
from my_utils.helpers import get_robust_fps

# Configure a logger for this module
logger = logging.getLogger(__name__)

def compute_iou(boxA, boxB):
    """
    Computes the Intersection over Union (IoU) of two bounding boxes.
    Each box is a tuple in the format (x1, y1, x2, y2).
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    if (boxAArea + boxBArea - interArea) <= 0:
        return 0
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

class YOLOCropper:
    def __init__(self, model_name="yolov5s", confidence_threshold=0.5):
        try:
            self.model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
            self.confidence_threshold = confidence_threshold
            self.selected_person_index = None
            self.last_selected_box = None
            self.smoothed_box = None
            logger.info(f"YOLOv5 model '{model_name}' loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load YOLOv5 model: {e}", exc_info=True)
            logger.error("Please ensure you have an internet connection for the first run, and that 'torch' and 'ultralytics' are installed.")
            raise

    def detect_persons(self, frame):
        """Run YOLO and return bounding boxes for detected persons."""
        results = self.model(frame)
        detections = results.xyxy[0].cpu().numpy()
        person_detections = [det for det in detections if int(det[5]) == 0 and det[4] >= self.confidence_threshold]
        boxes = [tuple(map(int, det[:4])) for det in person_detections]
        return boxes

    def prompt_user_for_selection(self, frame, boxes):
        """Display a window prompting the user to select a person."""
        display_frame = frame.copy()
        for idx, (x1, y1, x2, y2) in enumerate(boxes):
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Press {idx+1}"
            cv2.putText(display_frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        window_name = "Multiple persons detected - Select one"
        cv2.imshow(window_name, display_frame)
        logger.info("Multiple persons detected. Please press the number key corresponding to the person to crop.")

        key = cv2.waitKey(0) & 0xFF
        cv2.destroyWindow(window_name)
        cv2.waitKey(1)

        try:
            # Check for numeric keys '1' through '9'
            if ord('1') <= key <= ord('9'):
                selected_index = int(chr(key)) - 1
                if 0 <= selected_index < len(boxes):
                    logger.info(f"User selected person {selected_index + 1}.")
                    return selected_index
        except ValueError:
            pass # Key was not a number

        logger.warning("Invalid or no selection. Defaulting to the first detected person (index 0).")
        return 0

    def crop_video(self, input_video_path, output_video_path, margin=80, max_fail_frames=20, smoothing_factor=0.9):
        """
        Crops a video ONLY IF multiple people are detected. Otherwise, creates a
        link/copy with the correct name.
        """
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video file: {input_video_path}")
            return None, None
        
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = get_robust_fps(input_video_path)

        # --- PASS 1: Analyze video to determine if cropping is necessary ---
        logger.info("Starting Pass 1: Analyzing video to determine number of people...")
        frame_boxes = []
        max_crop_width = 0
        max_crop_height = 0
        consecutive_fail_count = 0
        multiple_people_found = False # Flag to track if we need to crop
        
        self.last_selected_box = None
        self.smoothed_box = None

        while True:
            ret, frame = cap.read()
            if not ret: break

            boxes = self.detect_persons(frame)
            current_box = None

            if len(boxes) > 1:
                multiple_people_found = True

            if not boxes:
                if self.last_selected_box:
                    current_box = self.last_selected_box
                    consecutive_fail_count += 1
                else:
                    consecutive_fail_count += 1
            elif len(boxes) == 1:
                current_box = boxes[0]
                consecutive_fail_count = 0
            else: # Multiple people detected
                if self.last_selected_box:
                    ious = [compute_iou(self.last_selected_box, b) for b in boxes]
                    best_idx = np.argmax(ious)
                    if ious[best_idx] > 0.3:
                        current_box = boxes[best_idx]
                        self.selected_person_index = best_idx
                    else:
                        self.selected_person_index = self.prompt_user_for_selection(frame, boxes)
                        current_box = boxes[self.selected_person_index]
                else:
                    self.selected_person_index = self.prompt_user_for_selection(frame, boxes)
                    current_box = boxes[self.selected_person_index]
                consecutive_fail_count = 0
            
            if consecutive_fail_count >= max_fail_frames:
                logger.warning(f"Person not detected for {max_fail_frames} frames. Ending analysis pass.")
                break

            if current_box:
                if self.smoothed_box is None: self.smoothed_box = current_box
                
                self.smoothed_box = tuple(
                    int(smoothing_factor * prev + (1 - smoothing_factor) * curr)
                    for prev, curr in zip(self.smoothed_box, current_box)
                )
                
                x1, y1, x2, y2 = self.smoothed_box
                crop_x1 = max(x1 - margin, 0)
                crop_y1 = max(y1 - margin, 0)
                crop_x2 = min(x2 + margin, orig_width)
                crop_y2 = min(y2 + margin, orig_height)
                
                max_crop_width = max(max_crop_width, crop_x2 - crop_x1)
                max_crop_height = max(max_crop_height, crop_y2 - crop_y1)
                
                frame_boxes.append((crop_x1, crop_y1, crop_x2, crop_y2))
                self.last_selected_box = self.smoothed_box
            else:
                frame_boxes.append(None)

        # --- DECISION POINT: CROP OR COPY ---
        if not multiple_people_found:
            logger.info("Only one person detected throughout the video. Creating a direct link/copy instead of cropping.")
            cap.release()
            
            if os.path.lexists(output_video_path): # Use lexists for symlinks
                os.remove(output_video_path)
            
            try:
                # Create a symbolic link for efficiency
                os.symlink(os.path.abspath(input_video_path), output_video_path)
                logger.info(f"SUCCESS: Symbolic link created at: {output_video_path}")
            except (OSError, AttributeError):
                logger.warning("Symbolic link failed (check permissions/OS support). Falling back to a full file copy.")
                shutil.copy2(input_video_path, output_video_path)
                logger.info(f"SUCCESS: Full copy created at: {output_video_path}")
                
            return output_video_path, (orig_width, orig_height)

        # --- PROCEED WITH CROPPING IF MULTIPLE PEOPLE WERE FOUND ---
        logger.info("Multiple people were detected. Proceeding with video cropping.")
        if max_crop_width == 0 or max_crop_height == 0:
            logger.error("Multiple people were detected, but none could be consistently tracked. Cannot create a cropped video.")
            cap.release()
            return None, None

        # --- PASS 2: Create the new video with padding ---
        logger.info(f"Final crop dimensions will be {max_crop_width}x{max_crop_height}.")
        logger.info("Starting Pass 2: Writing new cropped video...")
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Rewind video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (max_crop_width, max_crop_height))

        for frame_idx, crop_box in enumerate(frame_boxes):
            ret, frame = cap.read()
            if not ret: break
            if crop_box is None: continue

            canvas = np.zeros((max_crop_height, max_crop_width, 3), dtype=np.uint8)
            x1, y1, x2, y2 = crop_box
            cropped_person = frame[y1:y2, x1:x2]
            crop_h, crop_w, _ = cropped_person.shape
            paste_x = (max_crop_width - crop_w) // 2
            paste_y = (max_crop_height - crop_h) // 2
            canvas[paste_y : paste_y + crop_h, paste_x : paste_x + crop_w] = cropped_person
            out_writer.write(canvas)

        cap.release()
        out_writer.release()
        logger.info(f"SUCCESS: Cropped video saved to: {output_video_path}")
        return output_video_path, (max_crop_width, max_crop_height)