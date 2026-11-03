from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes.documents import router as documents_router

app = FastAPI(
    title="LLM Integrated OCR for Medical Document Analysis and Profile Generation",
    description="Backend API for document preprocessing, OCR confidence scoring, fallback triggering, and structured profile generation.",
    version="0.1.0",
)

# Enable CORS for frontend integration in subsequent phases
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routes
app.include_router(documents_router, prefix="/api")


@app.get("/health", tags=["Health"])
async def health_check():
    """
    Truthful service health probe.
    Returns status and service version without exposing any sensitive or clinical data.
    """
    return {
        "status": "healthy",
        "service": "medical-document-analysis-api",
        "version": "0.1.0",
    }


@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "LLM Integrated OCR for Medical Document Analysis API is running.",
        "docs_url": "/docs",
        "health_url": "/health",
        "upload_url": "/api/documents/upload",
    }
