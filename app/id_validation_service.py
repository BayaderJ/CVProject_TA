"""
ID Photo Validation Service - Using OpenCV DNN Face Detector
Replaces RetinaFace to avoid TensorFlow conflicts
"""

import cv2
import numpy as np
import math
from pathlib import Path
from typing import Dict, Tuple, Optional
import tensorflow as tf
from tensorflow import keras


class IDPhotoValidator:
    """
    Validates ID photo requirements for Saudi ID photos
    Uses OpenCV DNN face detector instead of RetinaFace
    """
    
    def __init__(self, models_dir: str = "app/models"):
        """
        Initialize the validator
        
        Args:
            models_dir: Directory containing the .keras model files
        """
        self.models_dir = Path(models_dir)
        
        print("[ID VALIDATOR] Loading validation models...")
        
        # Load the three .keras models
        self.glasses_model = self._load_keras_model("glasses")
        self.female_model = self._load_keras_model("female")
        self.male_model = self._load_keras_model("male")
        
        print("[ID VALIDATOR] All models loaded successfully")
        
        # Initialize OpenCV face detector
        self._initialize_opencv_detector()
        
        # Thresholds
        self.glasses_threshold = 0.35
        self.hijab_threshold = 0.5
        self.ghutra_threshold = 0.5
        
        # Image size for models
        self.img_size = (224, 224)
    
    def _initialize_opencv_detector(self):
        """
        Initialize OpenCV DNN face detector
        Uses pre-trained models that come with OpenCV
        """
        try:
            print("[ID VALIDATOR] Initializing OpenCV face detector...")
            
            # Try to use OpenCV's DNN face detector (Caffe model)
            # These are lightweight and don't conflict with TensorFlow
            self.face_detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            
            # Also load eye detector for landmark detection
            self.eye_detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_eye.xml'
            )
            
            print("[ID VALIDATOR] ✓ OpenCV face detector initialized")
            
        except Exception as e:
            print(f"[ID VALIDATOR] ⚠️ Face detector initialization failed: {e}")
            self.face_detector = None
            self.eye_detector = None
    
    def _load_keras_model(self, model_type: str) -> keras.Model:
        """
        Load a complete .keras model file
        """
        model_path = self.models_dir / f"{model_type}_model.keras"
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                f"Expected location: app/models/{model_type}_model.keras"
            )
        
        print(f"[ID VALIDATOR] Loading {model_type} model from {model_path}...")
        
        try:
            model = keras.models.load_model(str(model_path), compile=False)
            print(f"✓ {model_type} model loaded successfully")
            return model
        except Exception as e:
            try:
                model = keras.models.load_model(str(model_path), compile=False, safe_mode=False)
                print(f"✓ {model_type} model loaded successfully (safe_mode=False)")
                return model
            except Exception as e2:
                raise RuntimeError(f"Failed to load {model_type} model: {str(e)}\n{str(e2)}")
    
    def _predict_single(self, model: keras.Model, img_bgr: np.ndarray) -> float:
        """
        Run prediction on a single image with manual preprocessing
        """
        img_resized = cv2.resize(img_bgr, self.img_size, interpolation=cv2.INTER_AREA)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_float = img_rgb.astype(np.float32)
        img_preprocessed = keras.applications.mobilenet_v3.preprocess_input(img_float)
        x = np.expand_dims(img_preprocessed, axis=0)
        
        try:
            output = model.predict(x, verbose=0)
            prob = float(output[0][0])
        except Exception as e:
            print(f"[DEBUG] Prediction failed: {e}")
            prob = 0.0
        
        return prob
    
    def _detect_face_opencv(self, img_bgr: np.ndarray) -> Tuple[Optional[dict], Optional[tuple]]:
        """
        Detect face using OpenCV Haar Cascade
        
        Returns:
            (landmarks_dict, facial_area_tuple)
            landmarks_dict: {"left_eye": (x,y), "right_eye": (x,y), ...}
            facial_area_tuple: [x1, y1, x2, y2]
        """
        if self.face_detector is None:
            return None, None
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 100),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        if len(faces) == 0:
            return None, None
        
        # Get largest face
        areas = [w * h for (x, y, w, h) in faces]
        best_idx = np.argmax(areas)
        x, y, w, h = faces[best_idx]
        
        # Convert to [x1, y1, x2, y2] format
        facial_area = [x, y, x + w, y + h]
        
        # Detect eyes within face region for landmarks
        face_roi_gray = gray[y:y+h, x:x+w]
        eyes = self.eye_detector.detectMultiScale(
            face_roi_gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(20, 20)
        )
        
        landmarks = {}
        
        if len(eyes) >= 2:
            # Sort eyes by x coordinate (left to right)
            eyes_sorted = sorted(eyes, key=lambda e: e[0])
            
            # Left eye (first in sorted list)
            ex1, ey1, ew1, eh1 = eyes_sorted[0]
            left_eye = (x + ex1 + ew1//2, y + ey1 + eh1//2)
            
            # Right eye (second in sorted list)
            ex2, ey2, ew2, eh2 = eyes_sorted[1]
            right_eye = (x + ex2 + ew2//2, y + ey2 + eh2//2)
            
            landmarks = {
                "left_eye": left_eye,
                "right_eye": right_eye,
                "nose": (x + w//2, y + int(h*0.6)),  # Estimate nose position
                "mouth_left": (x + int(w*0.35), y + int(h*0.8)),  # Estimate
                "mouth_right": (x + int(w*0.65), y + int(h*0.8))  # Estimate
            }
        elif len(eyes) == 1:
            # Only one eye detected - estimate the other
            ex, ey, ew, eh = eyes[0]
            eye_center = (x + ex + ew//2, y + ey + eh//2)
            
            # Assume eye is centered, estimate both
            eye_spacing = w // 3
            landmarks = {
                "left_eye": (x + w//2 - eye_spacing//2, y + h//3),
                "right_eye": (x + w//2 + eye_spacing//2, y + h//3),
                "nose": (x + w//2, y + int(h*0.6)),
                "mouth_left": (x + int(w*0.35), y + int(h*0.8)),
                "mouth_right": (x + int(w*0.65), y + int(h*0.8))
            }
        else:
            # No eyes detected - estimate all landmarks from face box
            landmarks = {
                "left_eye": (x + int(w*0.35), y + int(h*0.35)),
                "right_eye": (x + int(w*0.65), y + int(h*0.35)),
                "nose": (x + w//2, y + int(h*0.6)),
                "mouth_left": (x + int(w*0.35), y + int(h*0.8)),
                "mouth_right": (x + int(w*0.65), y + int(h*0.8))
            }
        
        return landmarks, facial_area
    
    def validate_photo(self, img_bgr: np.ndarray, gender: str) -> Dict:
        """
        Validate a photo for Saudi ID requirements
        
        Args:
            img_bgr: BGR image from OpenCV
            gender: "male" or "female"
        
        Returns:
            Dictionary with validation results
        """
        errors = []
        warnings = []
        
        # ===========================================
        # STEP 1: GLASSES DETECTION (WARNING ONLY)
        # ===========================================
        try:
            print(f"[VALIDATOR] Checking for glasses...")
            glasses_prob = self._predict_single(self.glasses_model, img_bgr)
            print(f"[VALIDATOR] Glasses probability: {glasses_prob:.2%}")
            
            if glasses_prob >= self.glasses_threshold:
                warnings.append(
                    f"Glasses may be detected (confidence: {glasses_prob:.1%}). "
                    "If you are not wearing glasses, you may ignore this warning. "
                    "Otherwise, please remove glasses and retake the photo."
                )
        except Exception as e:
            print(f"[WARNING] Glasses detection failed: {e}")
            glasses_prob = 0.0
            warnings.append("Glasses detection unavailable")
        
        # ===========================================
        # STEP 2: HEAD COVER DETECTION (BLOCKING)
        # ===========================================
        try:
            if gender.lower() == "female":
                print(f"[VALIDATOR] Checking for hijab...")
                headcover_prob = self._predict_single(self.female_model, img_bgr)
                headcover_type = "hijab"
                print(f"[VALIDATOR] Hijab probability: {headcover_prob:.2%}")
                
                if headcover_prob < self.hijab_threshold:
                    errors.append(
                        f"Hijab not detected (confidence: {headcover_prob:.1%}). "
                        "For Saudi ID photos, females must wear hijab. "
                        "Please wear hijab and retake the photo."
                    )
            
            elif gender.lower() == "male":
                print(f"[VALIDATOR] Checking for ghutra/shemagh...")
                headcover_prob = self._predict_single(self.male_model, img_bgr)
                headcover_type = "ghutra/shemagh"
                print(f"[VALIDATOR] Ghutra/Shemagh probability: {headcover_prob:.2%}")
                
                if headcover_prob < self.ghutra_threshold:
                    errors.append(
                        f"Ghutra/Shemagh not detected (confidence: {headcover_prob:.1%}). "
                        "For Saudi ID photos, males must wear traditional head cover. "
                        "Please wear ghutra/shemagh and retake the photo."
                    )
            
            else:
                errors.append("Invalid gender. Must be 'male' or 'female'.")
                headcover_prob = 0.0
                headcover_type = "unknown"
        
        except Exception as e:
            print(f"[ERROR] Head cover detection failed: {e}")
            errors.append(f"Head cover detection failed: {str(e)}")
            headcover_prob = 0.0
            headcover_type = "unknown"
        
        # ===========================================
        # STEP 3: FACE DETECTION (OpenCV)
        # ===========================================
        try:
            print(f"[VALIDATOR] Running OpenCV face detection...")
            print(f"[VALIDATOR] Image shape: {img_bgr.shape}, dtype: {img_bgr.dtype}")
            
            landmarks, facial_area = self._detect_face_opencv(img_bgr)
            
            if landmarks is None or facial_area is None:
                print(f"[VALIDATOR] No face detected")
                errors.append(
                    "No face detected in image. Please ensure:\n"
                    "1. Face is clearly visible and well-lit\n"
                    "2. Face is not too small or too large\n"
                    "3. Image quality is good (not blurry)\n"
                    "4. Face is looking towards camera (frontal view)\n"
                    "5. Image size is at least 640x640 pixels"
                )
                head_tilt = 0
            else:
                print(f"[VALIDATOR] Face detected successfully")
                print(f"[VALIDATOR] Facial area: {facial_area}")
                
                # Calculate head tilt from eye positions
                left_eye = landmarks["left_eye"]
                right_eye = landmarks["right_eye"]
                
                # Ensure correct eye ordering
                if left_eye[0] > right_eye[0]:
                    left_eye, right_eye = right_eye, left_eye
                
                dx = right_eye[0] - left_eye[0]
                dy = right_eye[1] - left_eye[1]
                head_tilt = math.degrees(math.atan2(dy, dx))
                
                print(f"[VALIDATOR] Head tilt: {head_tilt:.2f}°")
        
        except Exception as e:
            print(f"[ERROR] Face detection failed: {e}")
            import traceback
            traceback.print_exc()
            errors.append(f"Face detection failed: {str(e)}")
            landmarks = None
            facial_area = None
            head_tilt = 0
        
        # ===========================================
        # RETURN VALIDATION RESULTS
        # ===========================================
        valid = len(errors) == 0
        
        result = {
            "valid": valid,
            "errors": errors,
            "warnings": warnings,
            "details": {
                "glasses_prob": glasses_prob,
                "headcover_prob": headcover_prob,
                "headcover_type": headcover_type,
                "gender": gender.lower(),
                "landmarks": landmarks,
                "facial_area": facial_area,
                "head_tilt": head_tilt
            }
        }
        
        if valid:
            print(f"[VALIDATOR] ✅ Validation passed")
        else:
            print(f"[VALIDATOR] ❌ Validation failed: {len(errors)} error(s)")
        
        return result


def get_validator() -> IDPhotoValidator:
    """
    Factory function to create and return the validator
    """
    return IDPhotoValidator()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python id_validation_service.py <image_path> <gender>")
        sys.exit(1)
    
    img_path = sys.argv[1]
    gender = sys.argv[2]
    
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not load image from {img_path}")
        sys.exit(1)
    
    validator = get_validator()
    result = validator.validate_photo(img, gender)
    
    print("\n" + "="*60)
    print("VALIDATION RESULTS")
    print("="*60)
    print(f"Valid: {result['valid']}")
    print(f"\nErrors ({len(result['errors'])}):")
    for error in result['errors']:
        print(f"  ❌ {error}")
    print(f"\nWarnings ({len(result['warnings'])}):")
    for warning in result['warnings']:
        print(f"  ⚠️  {warning}")
    print("="*60)