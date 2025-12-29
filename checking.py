#!/usr/bin/env python3
"""
Setup Verification Script
Run this after integration to verify everything is configured correctly
"""

import os
import sys
from pathlib import Path

def check_file_exists(filepath, required_size_mb=None):
    """Check if file exists and optionally verify size"""
    if not os.path.exists(filepath):
        return False, "Missing"
    
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    
    if required_size_mb and size_mb < required_size_mb:
        return False, f"Too small ({size_mb:.2f} MB, expected >{required_size_mb} MB)"
    
    return True, f"OK ({size_mb:.2f} MB)"

def main():
    print("="*70)
    print("SAUDI ID PHOTO SYSTEM - SETUP VERIFICATION")
    print("="*70)
    
    errors = []
    warnings = []
    
    # Check project structure
    print("\n[1] Checking project structure...")
    
    required_dirs = [
        "app",
        "app/models",
        "static",
        "experiments/pretrained_models",
        "gfpgan/weights"
    ]
    
    for dir_path in required_dirs:
        if os.path.isdir(dir_path):
            print(f"  ✅ {dir_path}")
        else:
            print(f"  ❌ {dir_path}")
            errors.append(f"Missing directory: {dir_path}")
    
    # Check model files
    print("\n[2] Checking model files...")
    
    model_files = [
        ("app/models/glasses_model.weights.h5", 3),
        ("app/models/glasses_config.json", None),
        ("app/models/glasses_metadata.json", None),
        ("app/models/female_model.weights.h5", 3),
        ("app/models/female_config.json", None),
        ("app/models/female_metadata.json", None),
        ("app/models/male_model.weights.h5", 3),
        ("app/models/male_config.json", None),
        ("app/models/male_metadata.json", None),
    ]
    
    for filepath, min_size in model_files:
        exists, status = check_file_exists(filepath, min_size)
        if exists:
            print(f"  ✅ {filepath} - {status}")
        else:
            print(f"  ❌ {filepath} - {status}")
            if filepath.endswith('.weights.h5'):
                errors.append(f"Model file issue: {filepath} - {status}")
            else:
                warnings.append(f"Missing metadata: {filepath}")
    
    # Check service files
    print("\n[3] Checking service files...")
    
    service_files = [
        "app/main.py",
        "app/attire_validation_service.py",
        "app/background_removal_service.py",
        "app/gfpgan_service.py",
        "app/id_validation_service.py"
    ]
    
    for filepath in service_files:
        if os.path.exists(filepath):
            print(f"  ✅ {filepath}")
        else:
            print(f"  ❌ {filepath}")
            errors.append(f"Missing service file: {filepath}")
    
    # Check Python dependencies
    print("\n[4] Checking Python dependencies...")
    
    dependencies = [
        ("fastapi", "FastAPI"),
        ("cv2", "OpenCV"),
        ("tensorflow", "TensorFlow"),
        ("PIL", "Pillow"),
        ("numpy", "NumPy"),
    ]
    
    for module_name, display_name in dependencies:
        try:
            __import__(module_name)
            print(f"  ✅ {display_name}")
        except ImportError:
            print(f"  ❌ {display_name}")
            errors.append(f"Missing dependency: {display_name} (pip install required)")
    
    # Check rembg separately (may need to download models)
    print("\n[5] Checking rembg (background removal)...")
    try:
        from rembg import remove
        print(f"  ✅ rembg installed")
        print(f"  ℹ️  Note: rembg will download ~200MB of models on first use")
    except ImportError:
        print(f"  ❌ rembg not installed")
        errors.append("Missing dependency: rembg (pip install rembg)")
    
    # Try to load models
    print("\n[6] Testing model loading...")
    
    try:
        sys.path.insert(0, os.path.abspath('.'))
        from app.attire_validation_service import get_attire_validator
        
        print("  Attempting to load attire validation models...")
        validator = get_attire_validator()
        print("  ✅ All attire validation models loaded successfully")
        
        # Test prediction on dummy data
        import numpy as np
        import cv2
        
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = validator.validate_photo(dummy_img, "male")
        print("  ✅ Model inference working")
        
    except FileNotFoundError as e:
        print(f"  ❌ Model file error: {e}")
        errors.append(f"Model loading failed: {e}")
    except Exception as e:
        print(f"  ❌ Model loading error: {e}")
        errors.append(f"Model loading failed: {e}")
    
    # Try to load background remover
    print("\n[7] Testing background removal...")
    
    try:
        from app.background_removal_service import get_background_remover
        
        print("  Attempting to initialize background remover...")
        bg_remover = get_background_remover()
        print("  ✅ Background remover initialized")
        
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        errors.append("Background removal not available - install rembg")
    except Exception as e:
        print(f"  ❌ Initialization error: {e}")
        warnings.append(f"Background removal issue: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    
    if not errors and not warnings:
        print("\n✅ ALL CHECKS PASSED! Your system is ready to use.")
        print("\nNext steps:")
        print("  1. Start the server: uvicorn app.main:app --reload")
        print("  2. Visit: http://localhost:8000")
        print("  3. Test with sample images")
        return 0
    
    if errors:
        print(f"\n❌ ERRORS FOUND ({len(errors)}):")
        for i, error in enumerate(errors, 1):
            print(f"  {i}. {error}")
    
    if warnings:
        print(f"\n⚠️  WARNINGS ({len(warnings)}):")
        for i, warning in enumerate(warnings, 1):
            print(f"  {i}. {warning}")
    
    print("\n" + "="*70)
    
    if errors:
        print("\n🔧 Please fix the errors above before running the application.")
        print("\nCommon fixes:")
        print("  - Missing model files: Re-download from Colab using MODEL_DOWNLOAD_INSTRUCTIONS.md")
        print("  - Missing dependencies: pip install -r requirements.txt")
        print("  - Missing service files: Copy files from integration package")
        return 1
    else:
        print("\n⚠️  System has warnings but should still work.")
        print("Consider fixing warnings for best performance.")
        return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)