import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
UPLOADS_DIR = BASE_DIR / "uploads"
ORIGINALS_DIR = UPLOADS_DIR / "originals"
PROCESSED_DIR = UPLOADS_DIR / "processed"

# Upload constraints
MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
ALLOWED_IMAGE_MIME_TYPES = {"image/png", "image/jpeg"}

# Preprocessing default parameters
# Local neighborhood size for Gaussian adaptive thresholding (must be an odd integer > 1)
GAUSSIAN_BLOCK_SIZE = 15
# Constant C subtracted from the weighted mean
GAUSSIAN_C = 10.0
# Deskew angle threshold: only rotate if absolute skew exceeds this threshold in degrees
MIN_DESKEW_ANGLE_DEG = 0.5
MAX_DESKEW_ANGLE_DEG = 45.0
