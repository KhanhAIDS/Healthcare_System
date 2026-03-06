"""
MediaPipe 0.10+ wrapper that provides a FaceMesh-like API for face landmark detection.
Uses the new FaceLandmarker task API.
"""

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from typing import NamedTuple, List
import os


class Landmark(NamedTuple):
    """Mimics MediaPipe's old Landmark structure"""
    x: float
    y: float
    z: float = 0.0


class FaceLandmarks(NamedTuple):
    """Container for face landmarks"""
    landmark: List[Landmark]


class FaceDetectionResult:
    """Wrapper for face detection results"""
    def __init__(self, multi_face_landmarks):
        self.multi_face_landmarks = multi_face_landmarks


class FaceMesh:
    """
    Wrapper around MediaPipe's FaceLandmarker (0.10+) that provides
    an API similar to the old solutions.face_mesh.FaceMesh
    """
    
    def __init__(self, static_image_mode=True, max_num_faces=1, 
                 refine_landmarks=True, min_detection_confidence=0.5):
        
        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.min_detection_confidence = min_detection_confidence
        self.refine_landmarks = refine_landmarks
        self.timestamp_ms = 0
        
        # Get model path
        model_path = "/workspace/face_landmarker.task"
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Face landmark model not found at {model_path}. "
                "Please download it using: python3 -c \"from mediapipe_wrapper import download_model; download_model()\""
            )
        
        # Determine running mode
        running_mode = vision.RunningMode.IMAGE if static_image_mode else vision.RunningMode.VIDEO

        # Create FaceLandmarker options
        options = vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=running_mode,
            num_faces=max_num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_detection_confidence,
            output_face_blendshapes=False, # Can be True if needed, but not used by old API
            output_facial_transformation_matrixes=False
        )
        
        # Create the FaceLandmarker
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def process(self, image):

        # Convert numpy array to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        
        # Run landmark detection based on mode
        if self.static_image_mode:
            detection_result = self.landmarker.detect(mp_image)
        else:
            self.timestamp_ms += 33
            detection_result = self.landmarker.detect_for_video(mp_image, self.timestamp_ms)
        
        # Convert results to old format
        if detection_result.face_landmarks:
            # Convert each face's landmarks
            face_landmarks_list = []
            for face_lms in detection_result.face_landmarks:
                # The new 0.10+ model returns 478 landmarks (468 face + 10 iris) by default.
                # If refine_landmarks was False in the old API, it expected only 468.
                final_lms = face_lms
                if not self.refine_landmarks and len(face_lms) > 468:
                     final_lms = face_lms[:468]

                # Convert NormalizedLandmark to Landmark format
                landmarks = [
                    Landmark(x=lm.x, y=lm.y, z=lm.z if hasattr(lm, 'z') else 0.0)
                    for lm in final_lms
                ]
                face_landmarks_list.append(FaceLandmarks(landmark=landmarks))
            
            return FaceDetectionResult(face_landmarks_list)
        
        return FaceDetectionResult(None)


def download_model():
    """Download the FaceLandmarker model if not present"""
    import urllib.request
    
    model_url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    model_path = "/workspace/face_landmarker.task"
    
    if not os.path.exists(model_path):
        print(f"Downloading FaceLandmarker model...")
        urllib.request.urlretrieve(model_url, model_path)
        print(f"✅ Model saved to {model_path}")
    else:
        print(f"✅ Model already exists at {model_path}")
