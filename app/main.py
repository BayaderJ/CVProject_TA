from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import io
import math
import os
import base64
from enum import Enum
from typing import Dict, Tuple, Optional
from groq import Groq
from .gfpgan_service import create_restorer
from .id_validation_service import get_validator
from .background_removal_service import get_background_remover

# Initialize Groq client for LLM
client = Groq(api_key="GROQ_API_KEY")

# LLM Configuration
default_layer_agent_config = {
    "layer_agent_1": {
        "system_prompt": (
            "أنت مساعد متخصص في معالجة الصور. اكتب تقريراً مختصراً جداً بالعربية.\n\n"
            "خطوات المعالجة:\n{helper_response}\n\n"
            "التعليمات المهمة:\n"
            "1. لا تكتب عناوين مثل 'تقرير نهائي' أو 'نتيجة التعديلات' - ابدأ مباشرة بالمحتوى\n"
            "2. اكتب فقط 2-3 جمل قصيرة عن التعديلات\n"
            "3. أضف نصيحة واحدة فقط مختصرة (جملة واحدة) عن تحسين الصور المستقبلية\n"
            "4. لا تستخدم النجوم ** أو أي رموز خاصة أو عناوين\n"
            "5. لا تستخدم نقاط أو قوائم - اكتب نصاً متصلاً فقط\n"
            "6. استخدم عربية واضحة فقط بدون أي رموز أو أحرف غريبة\n"
            "7. كن مختصراً جداً - 3 أسطر كحد أقصى"
        ),
        "model_name": "allam-2-7b",
        
        "temperature": 0.3
    }
}



def convert_to_json_serializable(obj):
    """Convert NumPy types to Python native types for JSON serialization"""
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_json_serializable(item) for item in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def generate_llm_report(log_steps: list) -> str:
    """
    Generating user-friendly report using LLM based on processing steps
    
    Args:
        log_steps: List of processing steps performed
        
    Returns:
        String containing the LLM-generated report in Arabic
    """
    try:
        helper_response = "\n".join(log_steps)
        
        for layer_name, cfg in default_layer_agent_config.items():
            response = client.chat.completions.create(
                model=cfg["model_name"],
                temperature=cfg["temperature"],
                messages=[
                    {
                        "role": "system", 
                        "content": cfg["system_prompt"].format(helper_response=helper_response)
                    },
                    {
                        "role": "user", 
                        "content": "اكتب تقريراً مختصراً جداً بدون عناوين أو رموز. فقط 2-3 جمل عن التعديلات ونصيحة واحدة قصيرة."
                    }
                ]
            )
            layer_output = response.choices[0].message.content
            # Clean up any remaining asterisks or special characters
            layer_output = layer_output.replace('**', '').replace('*', '')
            helper_response = layer_output
            final_report = layer_output
        
        return final_report
    except Exception as e:
        print(f"[LLM] Failed to generate report: {e}")
        # Fallback to simple concatenation if LLM fails
        return "\n".join(log_steps)


app = FastAPI()
# CORS middleware - Cross-Origin Resource Sharing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

print("[STARTUP] Loading models...")
restorer = create_restorer()
validator = get_validator()
bg_remover = get_background_remover()
print("[STARTUP] All models loaded successfully")

MIN_ROTATE = 2.0
MAX_AUTO_ROTATE = 20

SAUDI_ID_WIDTH_MM = 40
SAUDI_ID_HEIGHT_MM = 60
SAUDI_ID_DPI = 300

SAUDI_ID_WIDTH_PX = 480
SAUDI_ID_HEIGHT_PX = 640

FACE_SIZE_MIN = 0.70
FACE_SIZE_MAX = 0.80

JPEG_QUALITY = 95


def preprocess_for_detection(img: np.ndarray) -> np.ndarray:
    """Preprocess image for better face detection"""
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    
    h, w = img.shape[:2]
    print(f"[PREPROCESS] Original size: {w}x{h}")
    
    max_dim = 1920
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        print(f"[PREPROCESS] Resized to: {new_w}x{new_h}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_brightness = gray.mean()
    print(f"[PREPROCESS] Brightness: {mean_brightness:.1f}/255")
    
    if mean_brightness < 80:
        img = cv2.convertScaleAbs(img, alpha=1.2, beta=30)
        print(f"[PREPROCESS] Enhanced brightness (was too dark)")
    elif mean_brightness > 200:
        img = cv2.convertScaleAbs(img, alpha=0.9, beta=-20)
        print(f"[PREPROCESS] Reduced brightness (was too bright)")
    
    return img


class PhotoType(str, Enum):
    PROFESSIONAL = "professional"
    SAUDI_ID = "saudi_id"

class BackgroundColor(str, Enum):
    WHITE = "white"
    BLACK = "black"
    GREY = "grey"


@app.get("/", response_class=HTMLResponse)
async def home():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/api/process")
async def process_photo(
    file: UploadFile = File(...),
    photo_type: str = Form(...),
    background_color: str = Form(None),
    background_index: int = Form(None),
    gender: str = Form(None)
):
    """
    Main processing endpoint with separate pipelines for each photo type
    
    Args:
        file: Image file
        photo_type: "professional" or "saudi_id"
        background_color: "white", "black", "grey"
        background_index: Index of background image from backgrounds/ folder (optional)
        gender: "male" or "female" (required for saudi_id)
    """
    if photo_type not in ["professional", "saudi_id"]:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid photo type"}
        )
    
    if photo_type == "saudi_id":
        background_color = "white"
        background_index = None
        if not gender or gender.lower() not in ["male", "female"]:
            return JSONResponse(
                status_code=400,
                content={"error": "Gender (male/female) is required for Saudi ID photos"}
            )
    elif photo_type == "professional":
        if background_index is None and background_color not in ["white", "black", "grey", None]:
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid background color for professional photo"}
            )
    
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid image"}
        )
    
    try:
        if photo_type == "saudi_id":
            result = await process_saudi_id_photo(img, gender.lower())
        else:
            result = await process_professional_photo(img, background_color, background_index)
        
        return result
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Processing failed: {str(e)}"}
        )


async def process_professional_photo(img: np.ndarray, color: str = None, bg_index: int = None):
    """Professional photo pipeline:
    GFPGAN
    Background Removal
    Background Replacement
    LLM Report"""

    print(f"[PROFESSIONAL] Starting pipeline...")
    print(f"[PROFESSIONAL] Input: shape={img.shape}, dtype={img.dtype}")
    
    # Initialize processing log for LLM
    log_steps = []
    log_steps.append("بدء معالجة الصورة الاحترافية")
    
    try:
        print("[PROFESSIONAL] Step 1: Face restoration...")
        restored_img = apply_gfpgan_restoration(img)
        print(f"[PROFESSIONAL] Face restoration complete: shape={restored_img.shape}")
        log_steps.append("تم استعادة تفاصيل الوجه وتحسين الجودة باستخدام تقنية GFPGAN")
    except Exception as e:
        print(f"[PROFESSIONAL] Face restoration failed: {e}")
        return JSONResponse(
            status_code=400,
            content={"error": f"Face restoration failed: {str(e)}"}
        )
    
    try:
        print("[PROFESSIONAL] Step 2: Background removal and replacement...")
        if bg_index is not None:
            result_img = bg_remover.remove_and_replace(
                restored_img,
                bg_index=bg_index,
                use_professional_composite=True
            )
            log_steps.append(f"تم إزالة الخلفية الأصلية واستبدالها بخلفية احترافية")
        else:
            bg_color_tuple = get_bg_color(color if color else "white")
            print(f"[PROFESSIONAL] Using background color: {color} -> BGR{bg_color_tuple}")
            result_img = bg_remover.remove_and_replace(
                restored_img,
                bg_color=bg_color_tuple,
                use_professional_composite=True
            )
            color_names = {"white": "أبيض", "black": "أسود", "grey": "رمادي"}
            color_ar = color_names.get(color if color else "white", "أبيض")
            log_steps.append(f"تم إزالة الخلفية الأصلية واستبدالها بلون {color_ar} موحد")
        print(f"[PROFESSIONAL] Background replacement complete: shape={result_img.shape}")
    except Exception as e:
        print(f"[PROFESSIONAL] Background replacement failed: {e}")
        return JSONResponse(
            status_code=400,
            content={"error": f"Background replacement failed: {str(e)}"}
        )
    
    log_steps.append("تم تحسين جودة الصورة النهائية وضبط الإضاءة")
    
    # Generate LLM report
    print("[PROFESSIONAL] Generating LLM report...")
    llm_report = generate_llm_report(log_steps)
    
    print("[PROFESSIONAL] Step 3: Encoding final image as JPEG...")
    ok, encoded = cv2.imencode(".jpg", result_img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        return JSONResponse(
            status_code=500,
            content={"error": "Image encoding failed"}
        )
    
    # Encode image as base64 for JSON response
    image_base64 = base64.b64encode(encoded.tobytes()).decode('utf-8')
    
    print(f"[PROFESSIONAL] ✅ Pipeline complete! Output size: {len(encoded.tobytes())} bytes")
    
    # Return both image and LLM report
    return JSONResponse(
        content={
            "image_base64": image_base64,
            "report": llm_report,
            "processing_steps": log_steps
        }
    )


async def process_saudi_id_photo(img: np.ndarray, gender: str):
    """
    Complete Saudi ID pipeline with all validations and LLM report generation
    
    1. Glasses detection (warning only because our model has high false positive rate)
    2. Hijab/Ghutra detection (blocking - required)
    3. Face detection and positioning
    4. Head alignment correction
    5. Background removal and replacement (white)
    6. Resize to Saudi ID specifications (480×640px at 300 DPI)
    7. Generate LLM report
    
    Note: GFPGAN face restoration is DISABLED for Saudi ID to preserve original quality
    """
    errors = []
    warnings = []
    log_steps = []
    
    log_steps.append("بدء معالجة صورة الهوية السعودية")
    
    print(f"[SAUDI ID] Received image: shape={img.shape}, dtype={img.dtype}")
    
    if img is None or img.size == 0:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid or empty image"}
        )
    
    h, w = img.shape[:2]
    if h < 200 or w < 200:
        return JSONResponse(
            status_code=400,
            content={
                "error": f"Image too small ({w}x{h} pixels). Minimum size: 200x200 pixels. "
                "Please upload a higher resolution image."
            }
        )
    
    print(f"[SAUDI ID] Preprocessing image for face detection...")
    img = preprocess_for_detection(img)
    print(f"[SAUDI ID] After preprocessing: shape={img.shape}, dtype={img.dtype}")
    log_steps.append("تم تحسين جودة الصورة للكشف عن الوجه")
    
    print(f"[SAUDI ID] Starting validation for {gender} photo...")
    
    validation_result = validator.validate_photo(img, gender)
    validation_result = convert_to_json_serializable(validation_result)
    
    if not validation_result['valid']:
        return JSONResponse(
            status_code=400,
            content={
                "error": "Photo validation failed",
                "validation": {
                    "passed": False,
                    "errors": validation_result['errors'],
                    "warnings": validation_result['warnings'],
                    "details": validation_result['details']
                }
            }
        )
    
    warnings.extend(validation_result['warnings'])
    
    print(f"[SAUDI ID] Validation passed")
    print(f"  - Glasses prob: {validation_result['details']['glasses_prob']:.2%}")
    print(f"  - {validation_result['details']['headcover_type']} prob: {validation_result['details']['headcover_prob']:.2%}")
    
    # Add validation info to log
    headcover_type = "الحجاب" if gender == "female" else "الغترة"
    log_steps.append(f"تم التحقق من وجود {headcover_type} بنجاح")
    
    landmarks = validation_result['details'].get('landmarks')
    head_tilt = validation_result['details'].get('head_tilt', 0)
    face_bbox = validation_result['details'].get('facial_area')
    
    if landmarks is None or face_bbox is None:
        return JSONResponse(
            status_code=400,
            content={
                "error": "Could not detect face landmarks for processing",
                "validation": validation_result
            }
        )
    
    print(f"[SAUDI ID] Aligning face (tilt: {head_tilt:.2f}°)...")
    
    try:
        aligned_img, final_landmarks = align_face_if_needed(img, landmarks, head_tilt)
        if abs(head_tilt) >= MIN_ROTATE:
            log_steps.append(f"تم تصحيح ميل الرأس ({abs(head_tilt):.1f} درجة)")
    except ValueError as e:
        return JSONResponse(
            status_code=400,
            content={
                "error": str(e),
                "validation": validation_result
            }
        )
    except Exception as e:
        print(f"[WARNING] Alignment failed: {e}, continuing without alignment")
        aligned_img = img
        final_landmarks = landmarks
    
    print("[SAUDI ID] Skipping face restoration (preserving original quality)...")
    restored_img = aligned_img
    log_steps.append("تم الحفاظ على الجودة الأصلية للصورة")
    
    print("[SAUDI ID] Removing background...")
    
    try:
        img_with_new_bg = bg_remover.remove_and_replace(
            restored_img,
            bg_color=(255, 255, 255),
            use_professional_composite=True
        )
        log_steps.append("تم إزالة الخلفية واستبدالها بخلفية بيضاء موحدة")
    except Exception as e:
        print(f"[WARNING] Background removal failed: {e}, using original")
        img_with_new_bg = restored_img
        warnings.append(
            "Background removal failed - using original background. "
            "Please ensure photo has a plain background."
        )
    
    print(f"[SAUDI ID] Resizing to Saudi ID specifications ({SAUDI_ID_WIDTH_MM}×{SAUDI_ID_HEIGHT_MM}mm at {SAUDI_ID_DPI} DPI)...")
    
    final_img = resize_to_saudi_id_specs(img_with_new_bg)
    log_steps.append(f"تم تغيير حجم الصورة إلى المواصفات الرسمية ({SAUDI_ID_WIDTH_PX}×{SAUDI_ID_HEIGHT_PX} بكسل)")
    
    # Generate LLM report
    print("[SAUDI ID] Generating LLM report...")
    llm_report = generate_llm_report(log_steps)
    
    ok, encoded = cv2.imencode(".jpg", final_img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        return JSONResponse(
            status_code=500,
            content={"error": "Image encoding failed"}
        )
    
    # Encode image as base64
    image_base64 = base64.b64encode(encoded.tobytes()).decode('utf-8')
    
    print(f"[SAUDI ID] ✅ Processing complete!")
    print(f"  - Final size: {final_img.shape[1]}×{final_img.shape[0]} pixels")
    print(f"  - Warnings: {len(warnings)}")
    
    # Return JSON response with image and report
    return JSONResponse(
        content={
            "image_base64": image_base64,
            "report": llm_report,
            "processing_steps": log_steps,
            "warnings": warnings,
            "validation": {
                "passed": True,
                "details": validation_result['details']
            }
        }
    )


def resize_to_saudi_id_specs(img: np.ndarray) -> np.ndarray:

    #Resize to 480x640 while preserving face aspect ratio
    if img is None:
        raise ValueError("Input image is None")
    if not isinstance(img, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(img)}")
    if len(img.shape) != 3:
        raise ValueError(f"Expected 3D image, got shape {img.shape}")
    if img.dtype != np.uint8:
        raise TypeError(f"Expected uint8 dtype, got {img.dtype}")

    target_w = 480
    target_h = 640

    h, w = img.shape[:2]
    print(f"[RESIZE] Input: {w}x{h}")

    scale = min(target_w / w, target_h / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(
        img,
        (new_w, new_h),
        interpolation=cv2.INTER_LANCZOS4
    )

    canvas = np.full((target_h, target_w, 3), 255, dtype=np.uint8)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    adjusted = cv2.convertScaleAbs(canvas, alpha=1.05, beta=5)

    print(f"[RESIZE] Output: {target_w}x{target_h} (aspect ratio preserved)")
    return adjusted


def get_bg_color(color_name: str) -> Tuple[int, int, int]:
    #Converting color name to BGR tuple
    color_map = {
        "white": (255, 255, 255),
        "black": (0, 0, 0),
        "grey": (128, 128, 128)
    }
    return color_map.get(color_name, (255, 255, 255))


def align_face_if_needed(img_bgr: np.ndarray, landmarks: dict, angle_deg: float):
    """
    Align face if head tilt exceeds threshold
    
    Args:
        img_bgr: BGR image from OpenCV
        landmarks: Dictionary with 'left_eye', 'right_eye', etc.
        angle_deg: Pre-calculated head tilt angle
    
    Returns:
        (aligned_img, final_landmarks): Aligned image and updated landmarks
    
    Raises:
        ValueError: If tilt is too large (> MAX_AUTO_ROTATE)
    """
    left_eye = landmarks["left_eye"]
    right_eye = landmarks["right_eye"]
    
    if left_eye[0] > right_eye[0]:
        left_eye, right_eye = right_eye, left_eye
    
    abs_angle = abs(angle_deg)
    
    h, w = img_bgr.shape[:2]
    cx, cy = w // 2, h // 2
    
    if abs_angle < MIN_ROTATE:
        print(f"[ALIGNMENT] No rotation needed (tilt: {abs_angle:.2f}°)")
        return img_bgr, landmarks
    
    elif abs_angle > MAX_AUTO_ROTATE:
        raise ValueError(
            f"Head tilt is too large ({abs_angle:.1f}°). "
            f"Maximum allowed: {MAX_AUTO_ROTATE}°. Please retake the photo with head straight."
        )
    
    else:
        print(f"[ALIGNMENT] Rotating by {angle_deg:.2f}°...")
        
        M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
        
        aligned = cv2.warpAffine(
            img_bgr, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        try:
            from retinaface import RetinaFace
            
            faces = RetinaFace.detect_faces(aligned)
            if not faces:
                print("[WARNING] No face detected after alignment")
                return aligned, landmarks
            
            best_key = max(faces.keys(), key=lambda k: faces[k].get("score", 0))
            face = faces[best_key]
            final_landmarks = face["landmarks"]
            
            print("[ALIGNMENT] Face re-detected after rotation")
            return aligned, final_landmarks
            
        except Exception as e:
            print(f"[WARNING] Could not re-detect face: {e}")
            return aligned, landmarks


def apply_gfpgan_restoration(img: np.ndarray) -> np.ndarray:

    if img is None:
        raise ValueError("Input image is None")
    if not isinstance(img, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(img)}")
    if len(img.shape) != 3:
        raise ValueError(f"Expected 3D image, got shape {img.shape}")
    if img.shape[2] != 3:
        raise ValueError(f"Expected 3 channels (BGR), got {img.shape[2]}")
    if img.dtype != np.uint8:
        raise TypeError(f"Expected uint8 dtype, got {img.dtype}")
    
    print(f"[GFPGAN] Input: shape={img.shape}, dtype={img.dtype}")
    
    _, _, restored_img = restorer.enhance(
        img,
        has_aligned=False,
        only_center_face=False,
        paste_back=True,
        weight=0.5
    )
    
    if restored_img is None:
        raise ValueError("No face detected in image")
    
    if restored_img.dtype != np.uint8:
        restored_img = restored_img.astype(np.uint8)
    
    print(f"[GFPGAN] Output: shape={restored_img.shape}, dtype={restored_img.dtype}")
    
    return restored_img


@app.get("/api/backgrounds")
async def list_backgrounds():
    """List all available background images from backgrounds/ folder"""
    bg_files = bg_remover.list_available_backgrounds()
    
    return {
        "count": len(bg_files),
        "backgrounds": [
            {
                "index": i,
                "filename": os.path.basename(f),
                "path": f
            }
            for i, f in enumerate(bg_files)
        ]
    }


@app.post("/api/test-face-detection")
async def test_face_detection(file: UploadFile = File(...)):
    """Diagnostic endpoint to test face detection and provide detailed feedback"""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid image - could not decode"}
        )
    
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    
    diagnostics = {
        "image_info": {
            "width": w,
            "height": h,
            "shape": f"{w}x{h}",
            "dtype": str(img.dtype),
            "channels": img.shape[2] if len(img.shape) > 2 else 1,
            "size_check": "OK" if min(h, w) >= 200 else f"TOO SMALL (min 200px, got {min(h,w)}px)"
        },
        "brightness": {
            "mean": brightness,
            "min": int(img.min()),
            "max": int(img.max()),
            "assessment": "OK" if 80 <= brightness <= 200 else ("TOO DARK" if brightness < 80 else "TOO BRIGHT")
        }
    }
    
    try:
        from retinaface import RetinaFace
        
        print(f"[TEST] Attempting face detection on {w}x{h} image...")
        faces = RetinaFace.detect_faces(img)
        
        if not faces:
            diagnostics["detection_original"] = {
                "result": "NO_FACE_DETECTED",
                "suggestion": [
                    "Ensure face is clearly visible and well-lit",
                    "Face should occupy 30-60% of image width",
                    "Try better lighting conditions",
                    "Ensure image is not blurry",
                    "Face should be looking towards camera (not profile)",
                    "Try with image at least 640px wide"
                ]
            }
        else:
            diagnostics["detection_original"] = {
                "result": "SUCCESS",
                "num_faces": len(faces),
                "faces": []
            }
            for key, face in faces.items():
                face_info = {
                    "confidence": float(face.get("score", 0)),
                    "bbox": face["facial_area"],
                    "landmarks_detected": list(face["landmarks"].keys())
                }
                diagnostics["detection_original"]["faces"].append(face_info)
        
        if not faces:
            print("[TEST] Trying with preprocessing...")
            preprocessed = preprocess_for_detection(img.copy())
            faces_preprocessed = RetinaFace.detect_faces(preprocessed)
            
            if faces_preprocessed:
                diagnostics["detection_preprocessed"] = {
                    "result": "SUCCESS_AFTER_PREPROCESSING",
                    "num_faces": len(faces_preprocessed),
                    "message": "Face detected after preprocessing - image quality may be suboptimal"
                }
            else:
                diagnostics["detection_preprocessed"] = {
                    "result": "STILL_FAILED",
                    "message": "No face detected even after preprocessing"
                }
        
    except Exception as e:
        diagnostics["detection_error"] = {
            "error": str(e),
            "type": type(e).__name__
        }
    
    return JSONResponse(content=diagnostics)


@app.post("/api/validate-only")
async def validate_only(
    file: UploadFile = File(...),
    gender: str = Form(...)
):
    """Validation-only endpoint for Saudi ID photos"""
    if gender.lower() not in ["male", "female"]:
        return JSONResponse(
            status_code=400,
            content={"error": "Gender must be 'male' or 'female'"}
        )
    
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid image"}
        )
    
    img = preprocess_for_detection(img)
    
    validation_result = validator.validate_photo(img, gender.lower())
    validation_result = convert_to_json_serializable(validation_result)
    
    return {
        "validation": {
            "passed": validation_result['valid'],
            "errors": validation_result['errors'],
            "warnings": validation_result['warnings'],
            "details": validation_result['details'],
            "note": "Face size will be automatically adjusted during processing"
        }
    }


@app.post("/api/restore-only")
async def restore_only(file: UploadFile = File(...)):
    """Test endpoint for GFPGAN only"""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid image"}
        )
    
    try:
        restored_img = apply_gfpgan_restoration(img)
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": f"Face restoration failed: {str(e)}"}
        )
    
    ok, encoded = cv2.imencode(".jpg", restored_img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        return JSONResponse(
            status_code=500,
            content={"error": "Encoding failed"}
        )
    
    return StreamingResponse(
        io.BytesIO(encoded.tobytes()),
        media_type="image/jpeg",
        headers={
            "Content-Disposition": "inline; filename=restored_photo.jpg"
        }
    )


@app.post("/api/test-alignment")
async def test_alignment(file: UploadFile = File(...)):
    """Test endpoint for alignment only"""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid image"}
        )
    
    try:
        from retinaface import RetinaFace
        
        faces = RetinaFace.detect_faces(img)
        if not faces:
            return JSONResponse(
                status_code=400,
                content={"error": "No face detected"}
            )
        
        best_key = max(faces.keys(), key=lambda k: faces[k].get("score", 0))
        face = faces[best_key]
        landmarks = face["landmarks"]
        
        left_eye = landmarks["left_eye"]
        right_eye = landmarks["right_eye"]
        
        if left_eye[0] > right_eye[0]:
            left_eye, right_eye = right_eye, left_eye
        
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle_deg = math.degrees(math.atan2(dy, dx))
        
        aligned_img, final_landmarks = align_face_if_needed(img, landmarks, angle_deg)
        
        ok, encoded = cv2.imencode(".jpg", aligned_img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            return JSONResponse(
                status_code=500,
                content={"error": "Encoding failed"}
            )
        
        import json
        response_headers = {
            "Content-Disposition": "inline; filename=aligned_photo.jpg",
            "X-Original-Angle": str(angle_deg),
            "X-Alignment-Applied": str(abs(angle_deg) >= MIN_ROTATE)
        }
        
        return StreamingResponse(
            io.BytesIO(encoded.tobytes()),
            media_type="image/jpeg",
            headers=response_headers
        )
        
    except ValueError as e:
        return JSONResponse(
            status_code=400,
            content={"error": str(e)}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Alignment test failed: {str(e)}"}
        )


@app.post("/api/test-background-removal")
async def test_background_removal(file: UploadFile = File(...)):
    """Test endpoint for background removal only"""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid image"}
        )
    
    try:
        result_img = bg_remover.remove_and_replace(
            img,
            bg_color=(255, 255, 255),
            use_professional_composite=True
        )
        
        ok, encoded = cv2.imencode(".jpg", result_img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            return JSONResponse(
                status_code=500,
                content={"error": "Encoding failed"}
            )
        
        return StreamingResponse(
            io.BytesIO(encoded.tobytes()),
            media_type="image/jpeg",
            headers={
                "Content-Disposition": "inline; filename=bg_removed_photo.jpg"
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": f"Background removal failed: {str(e)}"}
        )

