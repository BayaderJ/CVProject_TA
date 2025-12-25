from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import io
from enum import Enum

from .gfpgan_service import create_restorer

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

# Create the GFPGAN model once at startup
restorer = create_restorer()

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
    background_color: str = Form(None)
):
    """
    Main processing endpoint with separate pipelines for each photo type
    """
    
    # Validate inputs
    if photo_type not in ["professional", "saudi_id"]:
        return {"error": "Invalid photo type"}
    
    # For Saudi ID, background must be white
    if photo_type == "saudi_id":
        background_color = "white"
    elif photo_type == "professional" and background_color not in ["white", "black", "grey"]:
        return {"error": "Invalid background color for professional photo"}
    
    # Read and decode image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return {"error": "Invalid image"}
    
    # STEP 1: Face Restoration (Same for both photo types - GFPGAN model)
    try:
        restored_img = apply_gfpgan_restoration(img)
    except Exception as e:
        return {"error": f"Face restoration failed: {str(e)}"}
    
    # STEP 2 & 3: Route to appropriate pipeline based on photo type
    try:
        if photo_type == "saudi_id":
            # Saudi ID Pipeline - Different measurements and specs
            final_img = process_saudi_id_pipeline(restored_img, background_color)
        else:  # professional
            # Professional Photo Pipeline - Standard processing
            final_img = process_professional_pipeline(restored_img, background_color)
    except Exception as e:
        return {"error": f"Processing failed: {str(e)}"}
    
    # Encode final result to PNG bytes
    ok, encoded = cv2.imencode(".png", final_img)
    if not ok:
        return {"error": "Image encoding failed"}
    
    return StreamingResponse(
        io.BytesIO(encoded.tobytes()),
        media_type="image/png",
    )


# ============================================================================
# STEP 1: GFPGAN FACE RESTORATION
# ============================================================================

def apply_gfpgan_restoration(img: np.ndarray) -> np.ndarray:
    """
    Apply GFPGAN face restoration
    This is YOUR implementation - works the same for both photo types
    
    Args:
        img: BGR image from OpenCV
    
    Returns:
        Restored BGR image
    """
    _, _, restored_img = restorer.enhance(
        img,
        has_aligned=False,
        only_center_face=False,
        paste_back=True,
        weight=0.5,
    )
    
    if restored_img is None:
        raise ValueError("No face detected in image")
    
    return restored_img


# ============================================================================
# PROFESSIONAL PHOTO PIPELINE
# ============================================================================

def process_professional_pipeline(img: np.ndarray, color: str) -> np.ndarray:
    """
    Complete pipeline for PROFESSIONAL photos (LinkedIn, Resume, etc.)
    
    Pipeline: GFPGAN → Background Removal → Background Replacement
    
    Args:
        img: BGR image after GFPGAN restoration
        color: Background color ('white', 'black', or 'grey')
    
    Returns:
        Final BGR image ready for download
    """
    print(f"[PROFESSIONAL PIPELINE] Processing with {color} background")
    
    # Step 2: Remove background
    img_no_bg = remove_background_professional(img)
    
    # Step 3: Replace background
    final_img = replace_background_professional(img_no_bg, color)
    
    return final_img


def remove_background_professional(img: np.ndarray) -> np.ndarray:
    """
    STEP 2 (Professional): Remove background from professional photo
    
    PLACEHOLDER to implement
    Professional photos have more flexible requirements
    
    Args:
        img: BGR image (from GFPGAN)
        Shape: (height, width, 3)
    
    Returns:
        BGRA image (with alpha channel for transparency)
        Shape: (height, width, 4)
        Alpha: 0 = background (transparent), 255 = person (opaque)
    
    TODO: Replace with actual background removal model
    Examples: rembg, U2-Net, DeepLabV3, etc.
    """
    print("[PLACEHOLDER - PROFESSIONAL] Background removal")
    
    # TODO: Implement your background removal model here
    # Example with rembg:
    # from rembg import remove
    # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # output = remove(img_rgb)
    # return cv2.cvtColor(output, cv2.COLOR_RGBA2BGRA)
    
    # Temporary placeholder: add alpha channel (fully opaque)
    if img.shape[2] == 3:
        alpha = np.ones((img.shape[0], img.shape[1], 1), dtype=img.dtype) * 255
        img_with_alpha = np.concatenate([img, alpha], axis=2)
        return img_with_alpha
    
    return img


def replace_background_professional(img: np.ndarray, color: str) -> np.ndarray:
    """
    STEP 3 (Professional): Replace background for professional photo
    
    PLACEHOLDER to implement
    Professional photos: standard background replacement
    
    Args:
        img: BGRA image (with alpha channel)
        Shape: (height, width, 4)
        color: 'white', 'black', or 'grey'
    
    Returns:
        BGR image with new background
        Shape: (height, width, 3)
    
    TODO: Replace with actual background replacement logic
    Can add edge refinement, color correction, etc.
    """
    print(f"[PLACEHOLDER - PROFESSIONAL] Background replacement with {color}")
    
    # Color mapping (BGR format for OpenCV)
    color_map = {
        "white": (255, 255, 255),
        "black": (0, 0, 0),
        "grey": (128, 128, 128)
    }
    
    bg_color = color_map.get(color, (255, 255, 255))
    
    # TODO: Implement your background replacement here
    # Can add edge feathering, color correction, etc.
    
    # Basic alpha blending (placeholder)
    if img.shape[2] == 4:
        background = np.full((img.shape[0], img.shape[1], 3), bg_color, dtype=np.uint8)
        alpha = img[:, :, 3:4].astype(float) / 255.0
        foreground = img[:, :, :3]
        final_img = (foreground * alpha + background * (1 - alpha)).astype(np.uint8)
        return final_img
    
    return img[:, :, :3] if img.shape[2] == 4 else img


# ============================================================================
# SAUDI ID PIPELINE (Different measurements and specifications)
# ============================================================================

def process_saudi_id_pipeline(img: np.ndarray, color: str) -> np.ndarray:
    """
    Complete pipeline for SAUDI ID photos
    
    Pipeline: GFPGAN → Background Removal (ID specs) → Background Replacement (ID specs)
    
    Saudi ID has specific requirements:
    - Specific dimensions (e.g., 4x6 cm or 413x531 pixels)
    - Face must be centered and sized properly
    - Specific contrast/brightness requirements
    - Always white background
    
    Args:
        img: BGR image after GFPGAN restoration
        color: Always 'white' for Saudi ID
    
    Returns:
        Final BGR image meeting Saudi ID specifications
    """
    print("[SAUDI ID PIPELINE] Processing with ID specifications")
    
    # Step 2: Remove background (with ID specifications)
    img_no_bg = remove_background_saudi_id(img)
    
    # Step 3: Replace background and apply ID specifications
    final_img = replace_background_saudi_id(img_no_bg, color)
    
    return final_img


def remove_background_saudi_id(img: np.ndarray) -> np.ndarray:
    """
    STEP 2 (Saudi ID): Remove background with ID photo specifications
    
    PLACEHOLDER to implement
    Saudi ID requirements are more strict than professional photos:
    - May need to detect and verify face position/size
    - Ensure proper head-to-shoulder ratio
    - Verify image meets ID standards before processing
    
    Args:
        img: BGR image (from GFPGAN)
        Shape: (height, width, 3)
    
    Returns:
        BGRA image (with alpha channel)
        Shape: (height, width, 4)
    
    Saudi ID Specifications:
    - Face should occupy 70-80% of image height
    - Head must be centered
    - Eyes should be at specific height (typically 2/3 from bottom)
    - No smile, neutral expression (already handled by photo capture)
    
    TODO: Replace with background removal model + ID validation
    """
    print("[PLACEHOLDER - SAUDI ID] Background removal with ID specs")
    
    # TODO: Implement Saudi ID specific background removal
    # 1. Verify face position and size meet ID requirements
    # 2. Remove background with higher precision (ID photos need clean edges)
    # 3. May need additional face detection to validate specs
    
    # Example structure:
    # - Detect face landmarks
    # - Verify face size ratio (head height / image height)
    # - Verify face is centered
    # - Remove background with stricter edge detection
    # - Return BGRA with clean alpha mask
    
    # Temporary placeholder
    if img.shape[2] == 3:
        alpha = np.ones((img.shape[0], img.shape[1], 1), dtype=img.dtype) * 255
        img_with_alpha = np.concatenate([img, alpha], axis=2)
        return img_with_alpha
    
    return img


def replace_background_saudi_id(img: np.ndarray, color: str) -> np.ndarray:
    """
    STEP 3 (Saudi ID): Replace background and apply ID specifications
    
    PLACEHOLDER to implement
    Saudi ID has strict requirements that differ from professional photos
    
    Args:
        img: BGRA image (with alpha channel)
        Shape: (height, width, 4)
        color: Always 'white' for Saudi ID
    
    Returns:
        BGR image meeting Saudi ID specifications
        Final dimensions: Typically 413x531 pixels (4x6 cm at 260 DPI)
    
    Saudi ID Specifications:
    - Exact dimensions: 4cm x 6cm (413 x 531 pixels at 260 DPI)
    - Background: Pure white (255, 255, 255)
    - Face centered with proper margins
    - Specific contrast and brightness levels
    - High quality, sharp edges
    
    TODO: Replace with Saudi ID specification implementation
    """
    print("[PLACEHOLDER - SAUDI ID] Background replacement with ID specs")
    
    # Saudi ID standard dimensions
    SAUDI_ID_WIDTH = 413   # 4 cm at 260 DPI
    SAUDI_ID_HEIGHT = 531  # 6 cm at 260 DPI
    
    # TODO: Implement Saudi ID specific processing:
    # 1. Replace background with pure white
    # 2. Resize to exact Saudi ID dimensions (413x531)
    # 3. Ensure face is properly centered
    # 4. Apply contrast/brightness adjustments for ID photos
    # 5. Ensure margins meet ID requirements
    
    # Color for Saudi ID (always white)
    bg_color = (255, 255, 255)  # Pure white in BGR
    
    # Basic implementation (placeholder)
    if img.shape[2] == 4:
        # Create white background
        background = np.full((img.shape[0], img.shape[1], 3), bg_color, dtype=np.uint8)
        alpha = img[:, :, 3:4].astype(float) / 255.0
        foreground = img[:, :, :3]
        blended = (foreground * alpha + background * (1 - alpha)).astype(np.uint8)
        
        # TODO: Resize to Saudi ID dimensions
        # final_img = cv2.resize(blended, (SAUDI_ID_WIDTH, SAUDI_ID_HEIGHT), interpolation=cv2.INTER_LANCZOS4)
        # TODO: Center face properly
        # TODO: Apply ID-specific adjustments
        
        return blended
    
    return img[:, :, :3] if img.shape[2] == 4 else img


# ============================================================================
# Test Endpoints for Individual Steps
# ============================================================================

@app.post("/api/restore-only")
async def restore_only(file: UploadFile = File(...)):
    """
    Test endpoint for GFPGAN only (for your testing)
    Tests Step 1 in isolation
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return {"error": "Invalid image"}
    
    restored_img = apply_gfpgan_restoration(img)
    
    ok, encoded = cv2.imencode(".png", restored_img)
    if not ok:
        return {"error": "Encoding failed"}
    
    return StreamingResponse(
        io.BytesIO(encoded.tobytes()),
        media_type="image/png",
    )


@app.post("/api/test-professional")
async def test_professional_pipeline(file: UploadFile = File(...), background_color: str = Form("white")):
    """
    Test endpoint for complete professional pipeline
    Useful for testing
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return {"error": "Invalid image"}
    
    # Skip GFPGAN for faster testing
    final_img = process_professional_pipeline(img, background_color)
    
    ok, encoded = cv2.imencode(".png", final_img)
    return StreamingResponse(io.BytesIO(encoded.tobytes()), media_type="image/png")


@app.post("/api/test-saudi-id")
async def test_saudi_id_pipeline(file: UploadFile = File(...)):
    """
    Test endpoint for complete Saudi ID pipeline
    Useful for testing
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return {"error": "Invalid image"}
    
    # Skip GFPGAN for faster testing
    final_img = process_saudi_id_pipeline(img, "white")
    
    ok, encoded = cv2.imencode(".png", final_img)
    return StreamingResponse(io.BytesIO(encoded.tobytes()), media_type="image/png")

