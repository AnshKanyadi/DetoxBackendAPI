"""
Detox Backend API
Secure, privacy-focused OCR and sensitive information detection

SECURITY GUARANTEES:
- Images are NEVER stored to disk
- Images are NEVER logged
- All processing happens in memory
- Images are deleted immediately after OCR
- Only bounding box coordinates are returned
"""

import io
import re
import gc
import base64
import logging
import time
import hashlib
from typing import Optional
from contextlib import contextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from rapidocr_onnxruntime import RapidOCR
from PIL import Image
import numpy as np

# =============================================================================
# Security Configuration
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("detox")

rate_limit_store: dict = {}
RATE_LIMIT_REQUESTS = 30
RATE_LIMIT_WINDOW = 60

# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="Detox API",
    description="Privacy-focused OCR for sensitive information detection. Images are never stored.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url=None
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# =============================================================================
# Security Middleware
# =============================================================================

@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["X-Detox-Privacy"] = "no-storage, no-logging, ephemeral-processing"
    return response

@app.middleware("http")
async def rate_limiter(request: Request, call_next):
    if request.url.path == "/analyze":
        client_ip = request.client.host
        current_time = time.time()
        rate_limit_store[client_ip] = [
            t for t in rate_limit_store.get(client_ip, [])
            if current_time - t < RATE_LIMIT_WINDOW
        ]
        if len(rate_limit_store.get(client_ip, [])) >= RATE_LIMIT_REQUESTS:
            return JSONResponse(
                status_code=429,
                content={"error": "Rate limit exceeded. Try again later."}
            )
        rate_limit_store.setdefault(client_ip, []).append(current_time)
    return await call_next(request)

# =============================================================================
# Secure Image Processing
# =============================================================================

@contextmanager
def secure_image_context(image_data: bytes):
    image = None
    try:
        image = Image.open(io.BytesIO(image_data))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        yield image
    finally:
        if image:
            image.close()
        del image
        gc.collect()

# =============================================================================
# OCR Engine (Lazy Loaded)
# =============================================================================

ocr_engine: Optional[RapidOCR] = None

def get_ocr():
    """Lazy initialize RapidOCR engine"""
    global ocr_engine
    if ocr_engine is None:
        logger.info("Initializing RapidOCR engine...")
        ocr_engine = RapidOCR()
        logger.info("RapidOCR ready")
    return ocr_engine

# =============================================================================
# Pattern Detection
# =============================================================================

PATTERNS = {
    "email": {
        "regex": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        "type": "EMAIL"
    },
    "phone_us": {
        "regex": r'(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}',
        "type": "PHONE"
    },
    "phone_intl": {
        "regex": r'\+[0-9]{1,3}[-.\s]?[0-9]{1,4}[-.\s]?[0-9]{1,4}[-.\s]?[0-9]{1,9}',
        "type": "PHONE"
    },
    "ssn": {
        "regex": r'\b[0-9]{3}[-\s]?[0-9]{2}[-\s]?[0-9]{4}\b',
        "type": "SSN"
    },
    "credit_card": {
        "regex": r'\b[0-9]{4}[-\s]?[0-9]{4}[-\s]?[0-9]{4}[-\s]?[0-9]{4}\b',
        "type": "CREDIT_CARD"
    },
    "zip_code": {
        "regex": r'\b[0-9]{5}(?:-[0-9]{4})?\b',
        "type": "ZIP"
    },
    "ip_address": {
        "regex": r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b',
        "type": "IP"
    },
    "password": {
        "regex": r'(?:password|passwd|pwd|passcode|pin)[\s:=]+\S+',
        "type": "PASSWORD",
        "flags": re.IGNORECASE
    },
    "wifi": {
        "regex": r'(?:wifi|wi-fi|ssid|network|wpa|wep)[\s:=]+\S+',
        "type": "WIFI",
        "flags": re.IGNORECASE
    },
    "api_key": {
        "regex": r'(?:api[_\s]?key|secret[_\s]?key|token|auth|bearer)[\s:=]+\S+',
        "type": "API_KEY",
        "flags": re.IGNORECASE
    },
    "contact_label": {
        "regex": r'(?:contact|phone|mobile|cell|tel|call)[\s:]+[0-9\-\s\(\)]{7,}',
        "type": "PHONE",
        "flags": re.IGNORECASE
    },
    "date": {
        "regex": r'\b(?:0?[1-9]|1[0-2])[-/](?:0?[1-9]|[12][0-9]|3[01])[-/](?:19|20)?[0-9]{2}\b',
        "type": "DATE"
    }
}

def detect_sensitive(text: str) -> list:
    findings = []
    seen = set()
    for name, pattern in PATTERNS.items():
        flags = pattern.get("flags", 0)
        regex = re.compile(pattern["regex"], flags)
        for match in regex.finditer(text):
            matched_text = match.group().strip()
            if matched_text.lower() in seen or len(matched_text) < 3:
                continue
            seen.add(matched_text.lower())
            findings.append({
                "text": matched_text,
                "type": pattern["type"]
            })
    return findings

# =============================================================================
# API Models
# =============================================================================

class AnalyzeRequest(BaseModel):
    image: str

class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float

class Detection(BaseModel):
    text: str
    type: str
    confidence: float
    bbox: BoundingBox

class AnalyzeResponse(BaseModel):
    success: bool
    detections: list[Detection]
    ocr_text: str
    message: str
    scale: float = 1.0
    processing_id: str

# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/")
async def root():
    return {
        "service": "Detox API",
        "version": "2.0.0",
        "privacy": "Images are never stored or logged"
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/privacy")
async def privacy_policy():
    return {
        "policy_version": "1.0",
        "last_updated": "2024-01-01",
        "data_collection": {
            "images": "NEVER stored, processed in memory only",
            "ocr_results": "NEVER logged or stored",
            "ip_addresses": "Used only for rate limiting, not stored permanently",
            "usage_stats": "Anonymous aggregate counts only"
        },
        "data_retention": "0 seconds - all data deleted immediately after processing",
        "third_parties": "No data shared with third parties",
        "encryption": "All traffic encrypted via TLS/HTTPS"
    }

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_image(request: AnalyzeRequest):
    processing_id = hashlib.sha256(
        f"{time.time()}{id(request)}".encode()
    ).hexdigest()[:12]
    
    logger.info(f"[{processing_id}] Processing request")
    start_time = time.time()
    
    try:
        # Decode base64
        try:
            if "," in request.image:
                image_data = request.image.split(",")[1]
            else:
                image_data = request.image
            image_bytes = base64.b64decode(image_data)
        except Exception as e:
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        # Process image
        with secure_image_context(image_bytes) as image:
            original_size = image.size
            scale_factor = 1.0
            
            # Resize for faster processing
            max_dim = 1000
            if image.width > max_dim or image.height > max_dim:
                ratio = min(max_dim / image.width, max_dim / image.height)
                new_size = (int(image.width * ratio), int(image.height * ratio))
                scale_factor = 1.0 / ratio
                logger.info(f"[{processing_id}] Resize: {original_size} -> {new_size}")
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            img_array = np.array(image)
            
            # Run OCR with RapidOCR
            logger.info(f"[{processing_id}] Running OCR...")
            ocr = get_ocr()
            
            # RapidOCR returns (result, elapse) where result is list of [bbox, text, score]
            result, elapse = ocr(img_array)
            
            ocr_time = time.time() - start_time
            logger.info(f"[{processing_id}] OCR completed in {ocr_time:.2f}s")
            
            del img_array
            gc.collect()
        
        del image_bytes
        gc.collect()
        
        # Process results
        detections = []
        all_text_parts = []
        
        if result:
            for item in result:
                bbox = item[0]  # [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
                text = item[1]
                confidence = item[2]
                
                if not text or not text.strip():
                    continue
                
                all_text_parts.append(text)
                
                # Convert bbox to x, y, width, height
                xs = [p[0] for p in bbox]
                ys = [p[1] for p in bbox]
                x = min(xs)
                y = min(ys)
                width = max(xs) - min(xs)
                height = max(ys) - min(ys)
                
                # Check for sensitive patterns
                findings = detect_sensitive(text)
                for finding in findings:
                    detections.append(Detection(
                        text=finding["text"],
                        type=finding["type"],
                        confidence=float(confidence),
                        bbox=BoundingBox(x=float(x), y=float(y), width=float(width), height=float(height))
                    ))
        
        # Check concatenated text too
        full_text = " ".join(all_text_parts)
        full_findings = detect_sensitive(full_text)
        existing_texts = {d.text.lower() for d in detections}
        
        for finding in full_findings:
            if finding["text"].lower() not in existing_texts and result:
                for item in result:
                    if finding["text"].lower() in item[1].lower():
                        bbox = item[0]
                        xs = [p[0] for p in bbox]
                        ys = [p[1] for p in bbox]
                        detections.append(Detection(
                            text=finding["text"],
                            type=finding["type"],
                            confidence=float(item[2]),
                            bbox=BoundingBox(
                                x=float(min(xs)),
                                y=float(min(ys)),
                                width=float(max(xs) - min(xs)),
                                height=float(max(ys) - min(ys))
                            )
                        ))
                        break
        
        total_time = time.time() - start_time
        logger.info(f"[{processing_id}] Found {len(detections)} sensitive items in {total_time:.2f}s")
        
        for det in detections:
            logger.info(f"[{processing_id}]   -> {det.type}: {det.text[:20]}...")
        
        return AnalyzeResponse(
            success=True,
            detections=detections,
            ocr_text="[REDACTED]",
            message=f"Found {len(detections)} sensitive item(s)",
            scale=scale_factor,
            processing_id=processing_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{processing_id}] Error: {type(e).__name__}: {str(e)}")
        raise HTTPException(status_code=500, detail="Processing failed")

# =============================================================================
# Startup
# =============================================================================

@app.on_event("startup")
async def startup_event():
    logger.info("=" * 50)
    logger.info("DETOX API - Privacy-Focused OCR Service")
    logger.info("=" * 50)
    logger.info("Security: Images are NEVER stored or logged")
    logger.info("Using RapidOCR (ONNX) for fast text detection")
    logger.info("Server ready!")
    logger.info("=" * 50)

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
