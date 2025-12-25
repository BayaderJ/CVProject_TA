# Integration Guide for Team Members

## 📋 Overview
This project has **TWO SEPARATE PIPELINES** for different photo types:
- **Professional Photos**: LinkedIn, Resume, Employee Cards (flexible specs)
- **Saudi ID Photos**: Official ID documents (strict specifications)

Both pipelines share Step 1 (GFPGAN) but have **different implementations** for Steps 2 & 3.

## 🏗️ Pipeline Architecture

### Overall Flow:
```
User Upload → GFPGAN (Step 1) → Route to Pipeline → Final Output
                                      ↓
                    ┌─────────────────┴─────────────────┐
                    ↓                                   ↓
            Professional Pipeline               Saudi ID Pipeline
            (Flexible specs)                    (Strict specs)
```

### Professional Photo Pipeline:
```
GFPGAN → remove_background_professional() → replace_background_professional() → Output
```

### Saudi ID Pipeline:
```
GFPGAN → remove_background_saudi_id() → replace_background_saudi_id() → Output
```

---

## 🔧 Where to Integrate Your Code

### Location: `app/main.py`

There are **FOUR placeholder functions** (2 for each pipeline):

---

## 📸 PROFESSIONAL PHOTO PIPELINE

### 1️⃣ Background Removal (Professional)

**Function:** `remove_background_professional()` (Line ~110)

**Purpose:** Remove background from professional photos (more flexible requirements)

**Input:**
- `img`: BGR image (from GFPGAN)
- Shape: `(height, width, 3)`

**Output:**
- BGRA image (with alpha channel)
- Shape: `(height, width, 4)`
- Alpha: 0 = background, 255 = person

**Example Implementation:**
```python
from rembg import remove

def remove_background_professional(img: np.ndarray) -> np.ndarray:
    """Remove background for professional photos"""
    
    # Convert BGR to RGB for rembg
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Remove background
    output = remove(img_rgb)
    
    # Convert RGBA to BGRA
    img_bgra = cv2.cvtColor(output, cv2.COLOR_RGBA2BGRA)
    
    return img_bgra
```

---

### 2️⃣ Background Replacement (Professional)

**Function:** `replace_background_professional()` (Line ~150)

**Purpose:** Replace background with chosen color (white/black/grey)

**Input:**
- `img`: BGRA image (with alpha)
- Shape: `(height, width, 4)`
- `color`: String - "white", "black", or "grey"

**Output:**
- BGR image with new background
- Shape: `(height, width, 3)`

**Example Implementation:**
```python
def replace_background_professional(img: np.ndarray, color: str) -> np.ndarray:
    """Replace background for professional photos"""
    
    color_map = {
        "white": (255, 255, 255),
        "black": (0, 0, 0),
        "grey": (128, 128, 128)
    }
    
    bg_color = color_map.get(color, (255, 255, 255))
    
    # Create solid background
    h, w = img.shape[:2]
    background = np.full((h, w, 3), bg_color, dtype=np.uint8)
    
    # Extract alpha and normalize
    alpha = img[:, :, 3:4].astype(float) / 255.0
    foreground = img[:, :, :3]
    
    # Alpha blending
    final = (foreground * alpha + background * (1 - alpha)).astype(np.uint8)
    
    return final
```

---

## 🆔 SAUDI ID PIPELINE (Different Specifications!)

### 3️⃣ Background Removal (Saudi ID)

**Function:** `remove_background_saudi_id()` (Line ~200)

**Purpose:** Remove background with **strict ID specifications**

**Saudi ID Requirements:**
- Face must occupy 70-80% of image height
- Head must be centered horizontally
- Eyes at 2/3 height from bottom
- Clean, precise edges (higher quality than professional)

**Input:**
- `img`: BGR image (from GFPGAN)
- Shape: `(height, width, 3)`

**Output:**
- BGRA image with precise alpha mask
- Shape: `(height, width, 4)`

**Example Implementation:**
```python
def remove_background_saudi_id(img: np.ndarray) -> np.ndarray:
    """Remove background with Saudi ID specifications"""
    
    # Step 1: Detect face to verify it meets ID specs
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.pth')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) == 0:
        raise ValueError("No face detected for Saudi ID")
    
    # Get largest face
    (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
    
    # Verify face size ratio (70-80% of image height)
    face_ratio = h / img.shape[0]
    if face_ratio < 0.65 or face_ratio > 0.85:
        print(f"Warning: Face size ratio {face_ratio:.2f} may not meet ID specs (should be 0.70-0.80)")
    
    # Verify face is centered (within 10% of center)
    face_center_x = x + w/2
    img_center_x = img.shape[1] / 2
    offset_ratio = abs(face_center_x - img_center_x) / img.shape[1]
    if offset_ratio > 0.1:
        print(f"Warning: Face not centered (offset: {offset_ratio:.2f})")
    
    # Step 2: Remove background with high precision
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    output = remove(img_rgb)  # or your preferred model
    
    # Step 3: Apply additional edge refinement for ID quality
    alpha = output[:, :, 3]
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)  # Slight smoothing
    output[:, :, 3] = alpha
    
    # Convert to BGRA
    img_bgra = cv2.cvtColor(output, cv2.COLOR_RGBA2BGRA)
    
    return img_bgra
```

---

### 4️⃣ Background Replacement (Saudi ID)

**Function:** `replace_background_saudi_id()` (Line ~270)

**Purpose:** Replace background and apply **Saudi ID specifications**

**Saudi ID Requirements:**
- **Exact dimensions**: 4cm × 6cm (413 × 531 pixels at 260 DPI)
- **Background**: Pure white (255, 255, 255)
- **Face positioning**: Centered with proper margins
- **Quality**: High contrast, sharp edges

**Input:**
- `img`: BGRA image
- Shape: `(height, width, 4)`
- `color`: Always "white" for Saudi ID

**Output:**
- BGR image with **exact Saudi ID dimensions**
- Shape: `(531, 413, 3)` - width=413, height=531

**Example Implementation:**
```python
def replace_background_saudi_id(img: np.ndarray, color: str) -> np.ndarray:
    """Apply Saudi ID specifications and background"""
    
    # Saudi ID standard dimensions (4cm x 6cm at 260 DPI)
    SAUDI_ID_WIDTH = 413
    SAUDI_ID_HEIGHT = 531
    
    # Step 1: Replace with pure white background
    bg_color = (255, 255, 255)
    h, w = img.shape[:2]
    background = np.full((h, w, 3), bg_color, dtype=np.uint8)
    
    alpha = img[:, :, 3:4].astype(float) / 255.0
    foreground = img[:, :, :3]
    blended = (foreground * alpha + background * (1 - alpha)).astype(np.uint8)
    
    # Step 2: Calculate scaling to fit Saudi ID dimensions
    # Preserve aspect ratio, then crop/pad to exact size
    current_h, current_w = blended.shape[:2]
    
    # Scale to fit height (face should fill most of the height)
    scale = SAUDI_ID_HEIGHT / current_h
    new_w = int(current_w * scale)
    new_h = SAUDI_ID_HEIGHT
    
    scaled = cv2.resize(blended, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    # Step 3: Center horizontally in Saudi ID dimensions
    if new_w > SAUDI_ID_WIDTH:
        # Crop from center
        start_x = (new_w - SAUDI_ID_WIDTH) // 2
        final_img = scaled[:, start_x:start_x + SAUDI_ID_WIDTH]
    else:
        # Pad with white on sides
        pad_left = (SAUDI_ID_WIDTH - new_w) // 2
        pad_right = SAUDI_ID_WIDTH - new_w - pad_left
        final_img = cv2.copyMakeBorder(
            scaled, 0, 0, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=bg_color
        )
    
    # Step 4: Apply contrast/brightness adjustments for ID photos
    # Saudi IDs typically need slightly higher contrast
    alpha_contrast = 1.1  # Contrast control (1.0-1.3)
    beta_brightness = 5   # Brightness control (0-50)
    final_img = cv2.convertScaleAbs(final_img, alpha=alpha_contrast, beta=beta_brightness)
    
    # Step 5: Ensure exact dimensions
    assert final_img.shape == (SAUDI_ID_HEIGHT, SAUDI_ID_WIDTH, 3), \
        f"Final image size mismatch: {final_img.shape}"
    
    return final_img
```

---

## 🧪 Testing Your Integration

### Test Professional Pipeline Only:
```bash
curl -X POST "http://localhost:8000/api/test-professional" \
  -F "file=@photo.jpg" \
  -F "background_color=white" \
  --output result_professional.png
```

### Test Saudi ID Pipeline Only:
```bash
curl -X POST "http://localhost:8000/api/test-saudi-id" \
  -F "file=@photo.jpg" \
  --output result_saudi_id.png
```

### Test Individual Steps:
```python
import cv2
import numpy as np
from app.main import remove_background_professional, remove_background_saudi_id

# Test professional background removal
img = cv2.imread("test.jpg")
result = remove_background_professional(img)
cv2.imwrite("test_pro_nobg.png", result)

# Test Saudi ID background removal
result = remove_background_saudi_id(img)
cv2.imwrite("test_saudi_nobg.png", result)
```

---

## 📊 Key Differences Between Pipelines

| Feature | Professional | Saudi ID |
|---------|-------------|----------|
| **Background Colors** | White, Black, Grey | White only |
| **Dimensions** | Flexible (original size) | Exact: 413×531 pixels |
| **Face Position** | Flexible | Strict: centered, 70-80% height |
| **Edge Quality** | Standard | High precision |
| **Contrast** | Normal | Slightly enhanced |
| **Validation** | Optional | Required (face detection) |

---

## 📦 Additional Dependencies for Saudi ID

Add to `requirements.txt`:

```txt
# For face detection/validation in Saudi ID
opencv-contrib-python==4.8.1.78  # Includes face detection models

# For better precision
scikit-image==0.22.0
```

---

## ⚠️ Common Issues

### Issue: Saudi ID face not detected
**Solution:** Ensure GFPGAN output has clear, frontal face

### Issue: Saudi ID dimensions wrong
**Solution:** Check your resize logic preserves aspect ratio first, then crops/pads

### Issue: Colors look different between pipelines
**Solution:** Professional and Saudi ID may have different contrast settings - this is intentional

---

## ✅ Integration Checklist

### For Professional Pipeline:
- [ ] `remove_background_professional()` implemented
- [ ] `replace_background_professional()` implemented
- [ ] Tested with white, black, and grey backgrounds
- [ ] Edge quality is acceptable

### For Saudi ID Pipeline:
- [ ] `remove_background_saudi_id()` implemented with validation
- [ ] `replace_background_saudi_id()` implemented with exact dimensions
- [ ] Face position/size validation works
- [ ] Output is exactly 413×531 pixels
- [ ] Background is pure white
- [ ] Contrast/brightness applied

---

## 💡 Pro Tips

1. **Reuse code where possible**: Both pipelines can share the same base background removal model, just with different validation/post-processing

2. **Test edge cases**: 
   - Very close faces
   - Faces at edge of frame
   - Multiple people (should reject for Saudi ID)

3. **Performance**: Load models once at module level, not in functions

4. **Error messages**: Be specific - help users fix their photos

---

## 🚀 Quick Start for Teammates

1. Find your functions in `app/main.py`:
   - Professional: Lines ~110 and ~150
   - Saudi ID: Lines ~200 and ~270

2. Replace placeholder code with your model

3. Test using the test endpoints

4. Verify with full pipeline

Good luck! 🎉