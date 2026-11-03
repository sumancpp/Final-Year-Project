from pydantic import BaseModel, Field


class ImageDimensions(BaseModel):
    width: int = Field(..., description="Image width in pixels")
    height: int = Field(..., description="Image height in pixels")
    channels: int = Field(..., description="Number of color channels (1 for grayscale/binary)")


class PreprocessingDetails(BaseModel):
    grayscale: bool = Field(True, description="Grayscale conversion applied")
    denoised: bool = Field(True, description="Noise reduction filter applied")
    adaptive_threshold: bool = Field(True, description="Gaussian adaptive thresholding applied")
    deskewed: bool = Field(..., description="Whether deskew rotation was performed")
    skew_angle_degrees: float = Field(..., description="Estimated or corrected skew angle in degrees")


class DocumentUploadResponse(BaseModel):
    document_id: str = Field(..., description="Unique server-generated identifier for the document")
    upload_status: str = Field("success", description="Status of the file upload and validation")
    original_filename: str = Field(..., description="Original name of the uploaded document (sanitized)")
    preprocessing_status: str = Field("completed", description="Status of the image preprocessing pipeline")
    dimensions: ImageDimensions = Field(..., description="Dimensions of the processed image")
    processed_document_id: str = Field(..., description="Safe identifier for retrieving or referencing processed image")
    preprocessing_details: PreprocessingDetails = Field(..., description="Details of image transformations executed")
