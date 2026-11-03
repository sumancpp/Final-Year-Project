# LLM Integrated OCR for Medical Document Analysis and Profile Generation

Final-year B.Tech Project — Computer Science & Engineering, Omdayal Group of Institutions.

---

## 1. Project Overview

Medical records, prescriptions, and lab reports often suffer from degradation, non-standard layouts, complex medical terminology, and varied handwriting. Standard OCR systems frequently stumble on degraded documents or messy handwriting, while running heavy Vision-Language Models (VLMs) on every document is computationally expensive and slow.

This project implements an **OCR-First Pipeline with Intelligent Vision LLM Fallback**. It leverages fast, deterministic Computer Vision and OCR/HTR as the primary extraction layer. An automated confidence assessment gate evaluates extraction certainty. When OCR confidence drops below a defined threshold or encounters ambiguous regions, a multimodal Vision LLM acts as a targeted semantic fallback to inspect visual context and reliably extract medical facts. The extracted entities are subsequently synthesized into a standardized, structured medical profile with patient record linking and chronological timeline generation.

---

## 2. Main System Workflow

```
[ Medical Document (Scan / Image / PDF) ]
                   │
                   ▼
       [ 1. Image Preprocessing ]
   • Grayscale conversion
   • Denoising (scanner/background noise reduction)
   • Gaussian Adaptive Thresholding (local neighborhood: T(x,y) = G(x,y) - C)
   • Deskewing and spatial normalization
                   │
                   ▼
     [ 2. OCR / HTR Primary Pass ]
   • Text Detection (locating lines and bounding boxes)
   • Feature Extraction & Recognition (visual patterns to tokens)
   • Sequence Decoding
                   │
                   ▼
       [ 3. Confidence Gate ]
   • Compute confidence metric (C_ocr)
        ├────────────────────────────┐
        ▼                            ▼
[ High Confidence: C_ocr ≥ τ ]  [ Weak / Uncertain: C_ocr < τ ]
        │                            │
        │                            ▼
        │               [ 4. Vision LLM Fallback ]
        │               • Multimodal image inspection
        │               • Resolves ambiguous/handwritten regions
        │                            │
        └─────────────┬──────────────┘
                      ▼
       [ 5. Medical Profile Generation ]
   • Structured entity extraction (Patient, Medications, Dosages, Schedules, Dates)
   • Strict fidelity: No invented values or hallucinated medical data
                      │
                      ▼
   [ 6. Record Linking & Chronological Timeline ]
   • Patient record correlation
   • Timeline synthesis across historical visits and prescriptions
```

---

## 3. Technology Stack

- **Backend Framework**: Python 3.11, FastAPI
- **Image Processing**: OpenCV (Grayscale, Denoising, Gaussian Adaptive Thresholding, Deskewing)
- **Document Ingestion**: Poppler (`pdftoppm`), PDF2Image, Pillow
- **OCR / HTR**: Modular OCR engine (Tesseract / PaddleOCR / EasyOCR)
- **Vision LLM & NLP**: Vision LLM fallback (API/Multimodal transformer), NLP entity structuring
- **Frontend**: React.js
- **Environment & Tools**: Debian GNU/Linux, Python `venv`, Git

---

## 4. Architecture & Design Principles

- **Modular Services**: Preprocessing, OCR, Confidence Evaluation, LLM Fallback, Profile Extraction, and Record Linking are isolated within dedicated service directories under `backend/app/services/`.
- **Provider Agnostic**: OCR engines and LLM providers are decoupled through standardized service interfaces, enabling model swaps without rewriting core application logic.
- **Data Integrity**: Medical data must maintain complete factual fidelity. Unclear, illegible, or absent parameters remain marked as unverified or absent rather than being guessed or hallucinated.

---

## 5. Directory Structure

```text
Final Year Project/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── routes/          # FastAPI API endpoint routers
│   │   ├── core/                # Application configuration and settings
│   │   ├── schemas/             # Pydantic schemas (request/response validation)
│   │   ├── services/            # Core business & processing services
│   │   │   ├── preprocessing/   # OpenCV filters, adaptive thresholding, deskewing
│   │   │   ├── ocr/             # Text detection, recognition, confidence calculation
│   │   │   ├── llm/             # Multimodal Vision LLM fallback integration
│   │   │   ├── profile/         # Medical profile structuring and extraction
│   │   │   └── record_linking/  # Cross-document linking and timeline generation
│   │   ├── models/              # Internal domain models / state definitions
│   │   └── utils/               # File helpers, logging, image conversion utilities
│   ├── uploads/                 # Local staging directory for uploaded documents
│   └── requirements.txt         # Backend Python dependencies
├── frontend/                    # React.js application
├── docs/
│   └── FinalYearProject.pdf     # Primary project specification
├── .venv/                       # Isolated Python 3.11 virtual environment
├── .gitignore                   # Version control ignore rules
└── README.md                    # Project documentation
```

---

## 6. Project Roadmap & Development Phases

- [x] **Phase 1: Environment Inspection & Preparation** *(Completed)*
- [x] **Phase 2: Project Folder Structure & Architecture** *(Completed)*
- [ ] **Phase 3: Backend Initialization & Dependency Installation** *(Next)*
- [ ] **Phase 4: Document Ingestion & OpenCV Preprocessing Pipeline**
- [ ] **Phase 5: OCR/HTR Engine Integration & Confidence Calculation**
- [ ] **Phase 6: Multimodal Vision LLM Fallback Service**
- [ ] **Phase 7: Medical Profile Extraction & Record Linking Engine**
- [ ] **Phase 8: FastAPI API Endpoints & Request Orchestration**
- [ ] **Phase 9: Frontend (React.js) UI Development**
- [ ] **Phase 10: End-to-End Integration & System Verification**
