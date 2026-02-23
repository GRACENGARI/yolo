# Use an official Python runtime as a parent image
# We use 3.10-slim for a balance of size and compatibility with ML libraries
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory
WORKDIR /app

# Install system dependencies required for OpenCV and build tools
RUN apt-get update && apt-get install -y 
    build-essential 
    cmake 
    libopenblas-dev 
    liblapack-dev 
    libx11-dev 
    libgtk-3-dev 
    libgl1-mesa-glx 
    libglib2.0-0 
    git 
    wget 
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .

# Install PyTorch with CUDA 11.8 support explicitly
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
# We install onnxruntime-gpu for CUDA support
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir insightface onnxruntime-gpu

# Copy the rest of the application code
COPY . .

# Expose the port the app runs on
EXPOSE 5001

# Command to run the application
# We'll override this in docker-compose or via CLI args
ENTRYPOINT ["python", "-m", "cv_engine.stream_processor"]
CMD ["--device", "cpu", "--source", "people.mp4"]
