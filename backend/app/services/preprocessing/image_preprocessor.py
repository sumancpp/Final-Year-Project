import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any

from backend.app.core.config import (
    GAUSSIAN_BLOCK_SIZE,
    GAUSSIAN_C,
    MIN_DESKEW_ANGLE_DEG,
    MAX_DESKEW_ANGLE_DEG,
    PROCESSED_DIR,
)


class ImagePreprocessingError(Exception):
    """Raised when image preprocessing fails or encounters corrupted data."""
    pass


class ImagePreprocessor:
    """
    Computer Vision Preprocessing Pipeline for Medical Documents.
    
    Implements:
    1. Grayscale intensity conversion
    2. Scanner noise reduction / denoising
    3. Gaussian Adaptive Thresholding (local neighborhood: T(x,y) = G(x,y) - C)
    4. Deskewing / orientation normalization
    """

    def __init__(
        self,
        block_size: int = GAUSSIAN_BLOCK_SIZE,
        c_constant: float = GAUSSIAN_C,
        min_deskew_angle: float = MIN_DESKEW_ANGLE_DEG,
        max_deskew_angle: float = MAX_DESKEW_ANGLE_DEG,
    ):
        # block_size must be an odd integer greater than 1
        if block_size % 2 == 0:
            block_size += 1
        self.block_size = max(3, block_size)
        self.c_constant = c_constant
        self.min_deskew_angle = min_deskew_angle
        self.max_deskew_angle = max_deskew_angle

    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert BGR/color image to 8-bit single-channel intensity image."""
        if len(image.shape) == 2:
            return image
        if len(image.shape) == 3:
            if image.shape[2] == 4:
                return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        raise ImagePreprocessingError("Unsupported image shape for grayscale conversion.")

    def denoise(self, gray_image: np.ndarray) -> np.ndarray:
        """
        Reduce scanner and background noise while preserving character edges.
        Uses Gaussian filter with a 3x3 kernel.
        """
        return cv2.GaussianBlur(gray_image, (3, 3), 0)

    def gaussian_adaptive_threshold(self, denoised_image: np.ndarray) -> np.ndarray:
        """
        Compute local pixel threshold using a Gaussian-weighted window.
        Mathematical Formulation:
          G(x,y) = sum(w_i * I_i)
          T(x,y) = G(x,y) - C
          B(x,y) = 255 if I(x,y) >= T(x,y) else 0
        """
        return cv2.adaptiveThreshold(
            denoised_image,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=self.block_size,
            C=self.c_constant,
        )

    def calculate_skew_angle(self, binary_image: np.ndarray) -> float:
        """
        Estimate rotation skew angle of text lines using minimum bounding area
        of foreground text pixels.
        """
        # Invert: text becomes foreground (white = 255), background becomes 0
        inverted = cv2.bitwise_not(binary_image)
        coords = np.column_stack(np.where(inverted > 0))

        if len(coords) < 50:
            # Insufficient text pixels to reliably detect skew
            return 0.0

        # Note: cv2.minAreaRect expects coordinates as (x, y) = (col, row)
        points = np.fliplr(coords)
        rect = cv2.minAreaRect(points)
        angle = rect[-1]

        # Normalize angle according to OpenCV conventions
        if angle < -45.0:
            angle = -(90.0 + angle)
        elif angle > 45.0:
            angle = 90.0 - angle
        else:
            angle = -angle

        return float(angle)

    def deskew(self, image: np.ndarray, angle: float) -> Tuple[np.ndarray, bool]:
        """
        Rotate image to correct text skew if the detected angle exceeds the threshold.
        Fills border with white (255) to maintain document background integrity.
        """
        if not (self.min_deskew_angle <= abs(angle) <= self.max_deskew_angle):
            return image, False

        height, width = image.shape[:2]
        center = (width // 2, height // 2)

        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
        rotated = cv2.warpAffine(
            image,
            rotation_matrix,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255,
        )
        return rotated, True

    def process_image_bytes(
        self,
        image_bytes: bytes,
        document_id: str,
    ) -> Dict[str, Any]:
        """
        Execute full preprocessing pipeline on raw image bytes and save output.
        
        Returns:
            Dict containing processing status, output path, dimensions, and transformation details.
        """
        # 1. Decode raw bytes to OpenCV BGR image
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)

        if image is None:
            raise ImagePreprocessingError("Failed to decode image data. File may be corrupted.")

        # 2. Grayscale conversion
        gray = self.to_grayscale(image)

        # 3. Denoising
        denoised = self.denoise(gray)

        # 4. Gaussian Adaptive Thresholding
        thresholded = self.gaussian_adaptive_threshold(denoised)

        # 5. Deskewing
        skew_angle = self.calculate_skew_angle(thresholded)
        deskewed_image, was_rotated = self.deskew(thresholded, skew_angle)

        # 6. Save processed image
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        processed_filename = f"{document_id}.png"
        processed_file_path = PROCESSED_DIR / processed_filename

        success = cv2.imwrite(str(processed_file_path), deskewed_image)
        if not success:
            raise ImagePreprocessingError("Failed to write processed image to disk.")

        height, width = deskewed_image.shape[:2]
        channels = 1 if len(deskewed_image.shape) == 2 else deskewed_image.shape[2]

        return {
            "processed_file_path": processed_file_path,
            "processed_filename": processed_filename,
            "dimensions": {
                "width": int(width),
                "height": int(height),
                "channels": int(channels),
            },
            "preprocessing_details": {
                "grayscale": True,
                "denoised": True,
                "adaptive_threshold": True,
                "deskewed": was_rotated,
                "skew_angle_degrees": round(skew_angle, 2),
            },
        }


# Singleton preprocessor instance
default_preprocessor = ImagePreprocessor()
