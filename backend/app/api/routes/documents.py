import re
import uuid
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, status

from backend.app.core.config import (
    MAX_UPLOAD_SIZE_BYTES,
    ALLOWED_IMAGE_EXTENSIONS,
    ALLOWED_IMAGE_MIME_TYPES,
    ORIGINALS_DIR,
)
from backend.app.schemas.document import DocumentUploadResponse, ImageDimensions, PreprocessingDetails
from backend.app.services.preprocessing import default_preprocessor, ImagePreprocessingError

router = APIRouter(prefix="/documents", tags=["Documents"])


def sanitize_filename(filename: str) -> str:
    """Strip directory paths and unsafe characters to prevent path traversal."""
    name = Path(filename).name
    # Keep alphanumeric characters, dots, underscores, and dashes
    sanitized = re.sub(r"[^\w.\-]", "_", name)
    return sanitized or "document.image"


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and preprocess a medical document image",
    description="Accepts a PNG or JPEG medical document, validates file integrity, and applies grayscale conversion, denoising, Gaussian adaptive thresholding, and deskewing.",
)
async def upload_document(
    file: UploadFile = File(..., description="Medical document scan or prescription image (PNG, JPEG, JPG)")
):
    # 1. Validate file extension
    raw_filename = file.filename or "unknown"
    sanitized_name = sanitize_filename(raw_filename)
    extension = Path(sanitized_name).suffix.lower()

    if extension not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file extension '{extension}'. Allowed extensions: {', '.join(sorted(ALLOWED_IMAGE_EXTENSIONS))}",
        )

    # 2. Validate MIME type
    content_type = (file.content_type or "").lower()
    if content_type not in ALLOWED_IMAGE_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported MIME type '{content_type}'. Allowed types: {', '.join(sorted(ALLOWED_IMAGE_MIME_TYPES))}",
        )

    # 3. Read content with size constraint enforcement (max 10 MB)
    content = await file.read(MAX_UPLOAD_SIZE_BYTES + 1)
    if len(content) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum upload size limit of {MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)} MB.",
        )

    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    # 4. Generate unique document ID and secure storage paths
    document_id = str(uuid.uuid4())
    ORIGINALS_DIR.mkdir(parents=True, exist_ok=True)
    stored_original_filename = f"{document_id}{extension}"
    original_file_path = ORIGINALS_DIR / stored_original_filename

    # 5. Save verified original raw bytes
    try:
        with open(original_file_path, "wb") as f:
            f.write(content)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to persist uploaded document.",
        )

    # 6. Execute OpenCV Preprocessing Pipeline (Grayscale, Denoising, Gaussian Adaptive Thresholding, Deskewing)
    try:
        result = default_preprocessor.process_image_bytes(
            image_bytes=content,
            document_id=document_id,
        )
    except ImagePreprocessingError as err:
        # Clean up original file if preprocessing decodes it as corrupted/invalid image
        if original_file_path.exists():
            original_file_path.unlink()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid or corrupted image content: {str(err)}",
        )
    except Exception:
        if original_file_path.exists():
            original_file_path.unlink()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during image preprocessing.",
        )

    # 7. Construct safe response without exposing server filesystem paths or sensitive clinical data
    return DocumentUploadResponse(
        document_id=document_id,
        upload_status="success",
        original_filename=sanitized_name,
        preprocessing_status="completed",
        dimensions=ImageDimensions(
            width=result["dimensions"]["width"],
            height=result["dimensions"]["height"],
            channels=result["dimensions"]["channels"],
        ),
        processed_document_id=result["processed_filename"],
        preprocessing_details=PreprocessingDetails(
            grayscale=result["preprocessing_details"]["grayscale"],
            denoised=result["preprocessing_details"]["denoised"],
            adaptive_threshold=result["preprocessing_details"]["adaptive_threshold"],
            deskewed=result["preprocessing_details"]["deskewed"],
            skew_angle_degrees=result["preprocessing_details"]["skew_angle_degrees"],
        ),
    )
