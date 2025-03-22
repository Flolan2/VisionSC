import matplotlib.pyplot as plt
import numpy as np
import cv2  # For video writing
import skvideo.io
import mediapipe as mp
import os
import pandas as pd
from tqdm import tqdm

from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions

from .mediapipe_landmarks import prepare_empty_dataframe


def load_models(min_hand_detection_confidence=0.5,
                min_tracking_confidence=0.5):
    """
    Load the hand and pose models using MediaPipe.
    Only the MultiIndex format is supported downstream.
    """
    CURRENT_DIR = os.path.dirname(__file__)
    HAND_MODEL = os.path.join(CURRENT_DIR, 'models', 'poet_hand_landmarker.task')
    POSE_MODEL = os.path.join(CURRENT_DIR, 'models', 'poet_pose_landmarker_full.task')

    VisionRunningMode = mp.tasks.vision.RunningMode
    
    # Hand detection
    base_options = mp.tasks.BaseOptions(model_asset_path=HAND_MODEL)
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    options = HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        running_mode=VisionRunningMode.VIDEO
    )
    options.min_hand_detection_confidence = min_hand_detection_confidence
    options.min_tracking_confidence = min_tracking_confidence
    hands = HandLandmarker.create_from_options(options)
    
    # Pose detection
    base_options = mp.tasks.BaseOptions(model_asset_path=POSE_MODEL)
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    options = PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=VisionRunningMode.VIDEO,
        min_pose_detection_confidence=0.8
    )
    pose = PoseLandmarker.create_from_options(options)
    
    return hands, pose
    

def track_video_list(video_list, 
                     output_folder='./tracking/',
                     overwrite=False, 
                     verbose=True, 
                     make_csv=True, 
                     make_video=True, 
                     world_coords=True,
                     min_tracking_confidence=0.7,
                     min_hand_detection_confidence=0.5):
    """
    Processes a list of videos and performs tracking on each,
    saving outputs only in the MultiIndex format.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for i, video in enumerate(video_list):
        # Reload models on each iteration (to reset timestamps)
        hands, pose = load_models(
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        video_name = os.path.basename(video).split('.')[0]
        
        if verbose:
            print(video)
        
        csv_path = os.path.join(output_folder, f"{video_name}_MPtracked.csv")
        if not overwrite and os.path.isfile(csv_path):
            if verbose:
                print('CSV file already exists - skipping this video.')
            continue
        
        track_video(
            video,
            pose,
            hands,
            output_folder=output_folder,
            make_csv=make_csv,
            make_video=make_video,
            world_coords=world_coords
        )
    
    return


def track_video(video, pose, hands,
                output_folder='./',
                make_csv=True,
                make_video=False,
                plot=False,
                world_coords=True):
    """
    Processes a single video for tracking and saves the output CSV and optionally an annotated video.
    This function only supports data with a MultiIndex column structure.
    """
    print(video)
    video_name = os.path.basename(video).split('.')[0]
    
    # Read video frames using skvideo.io.vreader
    videogen = list(skvideo.io.vreader(video))
    if not videogen:
        print("No frames read from video.")
        return
    
    # Get frame dimensions from the first frame
    frame_height, frame_width = videogen[0].shape[:2]
    
    # Extract FPS from video metadata (with a default fallback)
    metadata = skvideo.io.ffprobe(video)
    fps = 30  # default fallback
    if 'video' in metadata and '@r_frame_rate' in metadata['video']:
        fps_str = metadata['video']['@r_frame_rate']
        if '/' in fps_str:
            num, den = fps_str.split('/')
            try:
                fps = float(num) / float(den)
            except Exception:
                fps = 30
        else:
            try:
                fps = float(fps_str)
            except Exception:
                fps = 30
    
    writer_path = os.path.join(output_folder, f"{video_name}_MPtracked.mp4")
    
    # Initialize cv2.VideoWriter to save the annotated video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(writer_path, fourcc, fps, (frame_width, frame_height))
    
    # Create an empty DataFrame with MultiIndex columns
    marker_df, marker_mapping = prepare_empty_dataframe(hands='both', pose=True)
    
    for i, image in enumerate(tqdm(videogen, total=len(videogen))):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        # Calculate timestamp in milliseconds for the current frame
        frame_ms = int((i / fps) * 1000)
        
        # Perform Mediapipe predictions
        results_hands = hands.detect_for_video(mp_image, frame_ms)
        results_pose = pose.detect_for_video(mp_image, frame_ms)
        
        annotated_image = image.copy()
        
        # Determine image dimensions based on coordinate type
        if world_coords:
            img_h, img_w = (1, 1)
        else:
            img_h, img_w = image.shape[:2]

        # Process pose landmarks
        if results_pose.pose_world_landmarks:
            annotated_image = draw_pose_landmarks_on_image(annotated_image, results_pose)
            
            if world_coords:
                out = results_pose.pose_world_landmarks[0]
            else:
                out = results_pose.pose_landmarks[0]
            
            for l, landmark in enumerate(out):
                marker = marker_mapping['pose'][l]
                marker_df.loc[i, (marker, 'x')] = landmark.x * img_w
                marker_df.loc[i, (marker, 'y')] = landmark.y * img_h
                marker_df.loc[i, (marker, 'z')] = landmark.z
                marker_df.loc[i, (marker, 'visibility')] = landmark.visibility
                marker_df.loc[i, (marker, 'presence')] = landmark.presence

        # Process hand landmarks
        if results_hands.hand_landmarks:
            annotated_image = draw_hand_landmarks_on_image(annotated_image, results_hands)
            
            if world_coords:
                out = results_hands.hand_world_landmarks
            else:
                out = results_hands.hand_landmarks
            
            for h, hand in enumerate(out):
                handedness = results_hands.handedness[h][0].display_name
                for l, landmark in enumerate(hand):
                    marker = marker_mapping[handedness + '_hand'][l]
                    marker_df.loc[i, (marker, 'x')] = landmark.x * img_w
                    marker_df.loc[i, (marker, 'y')] = landmark.y * img_h
                    marker_df.loc[i, (marker, 'z')] = landmark.z
                    marker_df.loc[i, (marker, 'visibility')] = landmark.visibility
                    marker_df.loc[i, (marker, 'presence')] = landmark.presence

        if plot:
            plt.figure()
            plt.imshow(annotated_image)
        
        if make_video:
            writer.write(annotated_image)
    
    if make_csv:
        csv_path = os.path.join(output_folder, f"{video_name}_MPtracked.csv")
        # Ensure the DataFrame has a MultiIndex before saving
        if not isinstance(marker_df.columns, pd.MultiIndex):
            raise ValueError("The marker dataframe does not have a MultiIndex. This script only supports MultiIndex data.")
        marker_df.to_csv(csv_path)
    
    if make_video:
        writer.release()
    return


def draw_pose_landmarks_on_image(rgb_image, detection_result):
    """
    Annotate the image with pose landmarks.
    """
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)
    
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]
        
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
            for lm in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style()
        )
    return annotated_image


def draw_hand_landmarks_on_image(rgb_image, detection_result):
    """
    Annotate the image with hand landmarks.
    """
    hand_landmarks_list = detection_result.hand_landmarks
    annotated_image = np.copy(rgb_image)
    
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
            for lm in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style()
        )
    return annotated_image


def draw_face_landmarks_on_image(rgb_image, detection_result):
    """
    Annotate the image with face landmarks.
    """
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)
    
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]
        
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
            for lm in face_landmarks
        ])
        
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()
        )
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style()
        )
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style()
        )
    return annotated_image
