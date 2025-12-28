"""
Background Removal Service
Professional background removal using rembg with high-quality compositing
Based on CV_Before_Deploy.ipynb
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional


class BackgroundRemover:
    """
    Professional background removal service using rembg
    """
    
    def __init__(self):
        """
        Initialize the background remover
        """
        try:
            from rembg import remove
            self.remove_fn = remove
            print("[BACKGROUND REMOVER] rembg initialized successfully")
        except ImportError:
            raise ImportError(
                "rembg is not installed. Please install it:\n"
                "pip install rembg\n"
                "Note: This will download ~200MB of models on first use."
            )
    
    def remove_background(
        self, 
        img_bgr: np.ndarray,
        alpha_matting: bool = True,
        alpha_matting_foreground_threshold: int = 240,
        alpha_matting_background_threshold: int = 10,
        alpha_matting_erode_size: int = 10
    ) -> np.ndarray:
        """
        Remove background from image
        
        Args:
            img_bgr: BGR image from OpenCV
            alpha_matting: Use alpha matting for better edge quality
            alpha_matting_foreground_threshold: Foreground threshold (0-255)
            alpha_matting_background_threshold: Background threshold (0-255)
            alpha_matting_erode_size: Erosion kernel size for edge cleanup
        
        Returns:
            BGRA image (with alpha channel) - use this for compositing
        """
        # Convert BGR to RGB for rembg
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_img = Image.fromarray(img_rgb)
        
        # Remove background
        if alpha_matting:
            pil_result = self.remove_fn(
                pil_img,
                alpha_matting=True,
                alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
                alpha_matting_background_threshold=alpha_matting_background_threshold,
                alpha_matting_erode_size=alpha_matting_erode_size
            )
        else:
            pil_result = self.remove_fn(pil_img)
        
        # Convert back to numpy array (RGBA)
        img_rgba = np.array(pil_result)
        
        # Convert RGBA to BGRA for OpenCV
        img_bgra = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGRA)
        
        return img_bgra
    
    def professional_composite(
        self,
        fg_rgba: np.ndarray,
        bg_color: Tuple[int, int, int] = (255, 255, 255)
    ) -> np.ndarray:
        """
        Professional compositing with edge cleanup and smoothing
        Based on the notebook's professional_composite function
        
        Args:
            fg_rgba: RGBA foreground image (from rembg)
            bg_color: Background color as BGR tuple (default: white)
        
        Returns:
            BGR image with new background
        """
        # Convert RGBA to BGRA if needed
        if fg_rgba.shape[2] == 4:
            # Check if it's RGBA or BGRA
            # Assume it's BGRA (from our remove_background function)
            alpha = fg_rgba[:, :, 3]
            fg_bgr = fg_rgba[:, :, :3]
        else:
            raise ValueError("Input must be RGBA/BGRA image")
        
        # Edge cleanup: erode alpha to remove fringe pixels
        kernel = np.ones((3, 3), np.uint8)
        alpha = cv2.erode(alpha, kernel, iterations=1)
        
        # Edge smoothing: Gaussian blur on alpha
        alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
        
        # Create alpha mask (0-1 range)
        alpha_mask = alpha.astype(float) / 255.0
        alpha_mask = cv2.merge([alpha_mask, alpha_mask, alpha_mask])
        
        # Get dimensions
        th, tw = fg_rgba.shape[:2]
        
        # Create solid background
        background = np.full((th, tw, 3), bg_color, dtype=np.uint8).astype(float)
        
        # Convert foreground to float
        fg_float = fg_bgr.astype(float)
        
        # Alpha blending
        blended = (fg_float * alpha_mask) + (background * (1 - alpha_mask))
        
        # Clip and convert back to uint8
        result = np.clip(blended, 0, 255).astype(np.uint8)
        
        return result
    
    def remove_and_replace(
        self,
        img_bgr: np.ndarray,
        bg_color: Tuple[int, int, int] = (255, 255, 255),
        use_professional_composite: bool = True,
        alpha_matting: bool = True
    ) -> np.ndarray:
        """
        Remove background and replace with solid color in one step
        
        Args:
            img_bgr: Input BGR image
            bg_color: Background color as BGR tuple
            use_professional_composite: Use professional edge cleanup
            alpha_matting: Use alpha matting for removal
        
        Returns:
            BGR image with new background
        """
        # Remove background
        img_rgba = self.remove_background(
            img_bgr,
            alpha_matting=alpha_matting
        )
        
        # Composite with new background
        if use_professional_composite:
            result = self.professional_composite(img_rgba, bg_color)
        else:
            # Simple alpha blending
            alpha = img_rgba[:, :, 3:4].astype(float) / 255.0
            foreground = img_rgba[:, :, :3].astype(float)
            background = np.full(
                (img_rgba.shape[0], img_rgba.shape[1], 3),
                bg_color,
                dtype=float
            )
            result = (foreground * alpha + background * (1 - alpha)).astype(np.uint8)
        
        return result


def get_background_remover() -> BackgroundRemover:
    """
    Factory function to create and return the background remover
    
    Returns:
        BackgroundRemover instance
    """
    return BackgroundRemover()


# For testing
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python background_removal_service.py <image_path> [output_path]")
        print("  If output_path not provided, saves to 'output_nobg.png'")
        sys.exit(1)
    
    img_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "output_nobg.png"
    
    # Load image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not load image from {img_path}")
        sys.exit(1)
    
    print(f"Input image shape: {img.shape}")
    
    # Create remover
    remover = get_background_remover()
    
    # Test 1: Remove background only
    print("\n[TEST 1] Removing background (with alpha channel)...")
    img_rgba = remover.remove_background(img)
    cv2.imwrite("test_rgba.png", img_rgba)
    print(f"Saved RGBA image to test_rgba.png")
    
    # Test 2: Remove and replace with white
    print("\n[TEST 2] Remove and replace with white...")
    img_white = remover.remove_and_replace(img, bg_color=(255, 255, 255))
    cv2.imwrite(output_path, img_white)
    print(f"Saved white background to {output_path}")
    
    # Test 3: Remove and replace with other colors
    print("\n[TEST 3] Testing other background colors...")
    img_black = remover.remove_and_replace(img, bg_color=(0, 0, 0))
    cv2.imwrite("test_black_bg.png", img_black)
    print(f"Saved black background to test_black_bg.png")
    
    img_grey = remover.remove_and_replace(img, bg_color=(128, 128, 128))
    cv2.imwrite("test_grey_bg.png", img_grey)
    print(f"Saved grey background to test_grey_bg.png")
    
    print("\n✅ All tests completed successfully!")