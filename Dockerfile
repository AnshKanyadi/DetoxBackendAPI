# Detox API - Production Dockerfile
# Privacy-focused OCR service using EasyOCR

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CUDA_VISIBLE_DEVICES="" \
    EASYOCR_MODULE_PATH=/home/detox/.EasyOCR

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install numpy first, then CPU-only PyTorch
RUN pip install --no-cache-dir "numpy<2.0.0" && \
    pip install --no-cache-dir \
    torch==2.1.0+cpu \
    torchvision==0.16.0+cpu \
    -f https://download.pytorch.org/whl/cpu/torch_stable.html

# Install other dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .

# Create non-root user
RUN useradd --create-home --shell /bin/bash detox \
    && mkdir -p /home/detox/.EasyOCR \
    && chown -R detox:detox /home/detox

USER detox

EXPOSE 8000

# Health check with longer start period for model loading
HEALTHCHECK --interval=30s --timeout=30s --start-period=180s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
