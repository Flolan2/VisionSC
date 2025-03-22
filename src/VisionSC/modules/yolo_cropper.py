import cv2
import torch
import os

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
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

class YOLOCropper:
    def __init__(self, model_name="yolov5s", confidence_threshold=0.5):
        # Load the pre-trained YOLOv5 model from Ultralytics via torch.hub
        self.model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
        self.confidence_threshold = confidence_threshold
        self.selected_person_index = None  # user-selected person index (0-indexed)
        self.last_selected_box = None      # last known bounding box
        self.smoothed_box = None           # smoothed bounding box over frames

    def detect_persons(self, frame):
        """
        Run YOLO on the given frame and return a list of bounding boxes for detected persons.
        Each bounding box is a tuple: (x1, y1, x2, y2).
        """
        results = self.model(frame)
        detections = results.xyxy[0].cpu().numpy()
        # Class 0 corresponds to 'person'
        person_detections = [det for det in detections if int(det[5]) == 0 and det[4] >= self.confidence_threshold]
        boxes = []
        for det in person_detections:
            x1, y1, x2, y2 = map(int, det[:4])
            boxes.append((x1, y1, x2, y2))
        return boxes

    def prompt_user_for_selection(self, frame, boxes):
        """
        Draw bounding boxes with labels on the frame and display a window prompting the user
        to select one of the detected persons by pressing a number key.
        Returns the selected index (0-indexed).
        """
        display_frame = frame.copy()
        for idx, (x1, y1, x2, y2) in enumerate(boxes):
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display_frame, f"{idx+1}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        window_name = "Multiple persons detected - Press the corresponding number key"
        cv2.imshow(window_name, display_frame)
        print("Multiple persons detected. Please press the number key corresponding to the person to crop (e.g., '1' for first person).")
        
        key = cv2.waitKey(0)
        cv2.destroyWindow(window_name)
        cv2.waitKey(1)  # allow the window to close properly
        
        try:
            selected_index = int(chr(key)) - 1  # Convert pressed key to 0-indexed integer.
            if selected_index < 0 or selected_index >= len(boxes):
                print("Invalid selection. Defaulting to first detected person.")
                return 0
            return selected_index
        except Exception:
            print("Error in selection. Defaulting to first detected person.")
            return 0

    def crop_video(self, input_video_path, output_video_path, margin=80, max_fail_frames=20, smoothing_factor=0.9):
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {input_video_path}")
    
        # Option 1: Continue using OpenCV's method:
        # fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Option 2: Use robust FPS extraction from helpers:
        from my_utils.helpers import get_robust_fps
        fps = get_robust_fps(input_video_path)
        
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = None
        cropped_frame_size = None
    
        consecutive_fail_count = 0
    
        while True:
            ret, frame = cap.read()
            if not ret:
                break
    
            boxes = self.detect_persons(frame)
            selected_box = None
    
            if not boxes:
                if self.last_selected_box is not None:
                    selected_box = self.last_selected_box
                else:
                    selected_box = (0, 0, orig_width, orig_height)
                consecutive_fail_count += 1
            elif len(boxes) == 1:
                selected_box = boxes[0]
                consecutive_fail_count = 0
            else:
                if self.last_selected_box is not None:
                    best_iou = 0
                    best_box = None
                    best_idx = -1
                    for idx, box in enumerate(boxes):
                        iou = compute_iou(self.last_selected_box, box)
                        if iou > best_iou:
                            best_iou = iou
                            best_box = box
                            best_idx = idx
                    if best_iou > 0.3:
                        selected_box = best_box
                        self.selected_person_index = best_idx
                        consecutive_fail_count = 0
                    else:
                        self.selected_person_index = self.prompt_user_for_selection(frame, boxes)
                        selected_box = boxes[self.selected_person_index]
                        consecutive_fail_count = 0
                else:
                    self.selected_person_index = self.prompt_user_for_selection(frame, boxes)
                    selected_box = boxes[self.selected_person_index]
                    consecutive_fail_count = 0
    
            if consecutive_fail_count >= max_fail_frames:
                print(f"Person not detected for {max_fail_frames} consecutive frames. Cutting video here.")
                break
    
            if self.smoothed_box is None:
                self.smoothed_box = selected_box
            else:
                self.smoothed_box = tuple(
                    int(smoothing_factor * prev + (1 - smoothing_factor) * curr)
                    for prev, curr in zip(self.smoothed_box, selected_box)
                )
    
            x1, y1, x2, y2 = self.smoothed_box
            x1 = max(x1 - margin, 0)
            y1 = max(y1 - margin, 0)
            x2 = min(x2 + margin, orig_width)
            y2 = min(y2 + margin, orig_height)
            cropped_frame = frame[y1:y2, x1:x2]
    
            if out_writer is None:
                cropped_frame_size = (cropped_frame.shape[1], cropped_frame.shape[0])
                out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, cropped_frame_size)
            else:
                if (cropped_frame.shape[1], cropped_frame.shape[0]) != cropped_frame_size:
                    cropped_frame = cv2.resize(cropped_frame, cropped_frame_size)
            out_writer.write(cropped_frame)
            self.last_selected_box = selected_box
    
        cap.release()
        if out_writer:
            out_writer.release()
        print(f"Cropped video saved to: {output_video_path}")
        return output_video_path, cropped_frame_size

    