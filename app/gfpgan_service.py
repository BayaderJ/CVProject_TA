import os
from gfpgan import GFPGANer

# Base directory is CVPROJECT
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(
    BASE_DIR,
    "experiments",
    "pretrained_models",
    "GFPGANv1.3.pth",
)
def create_restorer():
    """
    Create a single GFPGAN v1.3 restorer
    Downloads model automatically if not present
    
    Returns:
        GFPGANer: Initialized GFPGAN model
    
    Raises:
        RuntimeError: If model download or initialization fails
    """
    # Auto-download if missing
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found. Downloading GFPGANv1.3.pth...")
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        
        import urllib.request
        MODEL_URL = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth"
        
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print(f"✓ Model downloaded to: {MODEL_PATH}")
        except Exception as e:
            raise RuntimeError(f"Failed to download model: {str(e)}")
    
    try:
        print(f"Loading GFPGAN model from: {MODEL_PATH}")
        
        restorer = GFPGANer(
            model_path=MODEL_PATH,
            upscale=2,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
        )
        
        print("✓ GFPGAN model loaded successfully")
        return restorer
        
    except Exception as e:
        raise RuntimeError(f"Failed to initialize GFPGAN model: {str(e)}")
