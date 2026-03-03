# HAWKEYE CV-Engine Docker Image
# Supports both CPU and GPU deployment with Phase 4 features
FROM python:3.10-slim

# Build arguments for flexibility
ARG DEVICE=cpu
ARG PYTORCH_VERSION=2.0.0

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV DEVICE=${DEVICE}

# Set the working directory
WORKDIR /app

# Install system dependencies required for OpenCV and ML libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY cv_engine/requirements.txt .

# Install PyTorch (CPU or CUDA based on build arg)
RUN if [ "$DEVICE" = "cuda" ] || [ "$DEVICE" = "gpu" ]; then \
        pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118; \
    else \
        pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu; \
    fi

# Install core dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional ML libraries
RUN pip install --no-cache-dir \
    insightface \
    onnxruntime-gpu \
    torchreid \
    gfpgan \
    basicsr \
    facexlib

# Copy the application code
COPY . .

# Create directories for weights, forensic output, and logs
RUN mkdir -p /app/weights /app/forensic_audit /app/logs /app/training_data

# Download default model weights (optional, can be mounted as volume)
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" || true

# Expose the port for MJPEG stream
EXPOSE 5001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:5001/video_feed || exit 1

# Command to run the application
ENTRYPOINT ["python", "-m", "cv_engine.stream_processor"]
CMD []
