# Use PyTorch CUDA image as base for GPU support
FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime AS base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    libc6 \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install git+https://github.com/openai/CLIP.git
RUN pip install git+https://github.com/ChaoningZhang/MobileSAM.git

# Copy application code 
COPY . .

RUN mkdir -p /root/.cache/clip
COPY weights/ /root/.cache/clip/

# Create necessary directories
RUN mkdir -p uploads weights

# Expose port
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=app.py
ENV PYTHONUNBUFFERED=1
ENV NUMPY_EXPERIMENTAL_ARRAY_FUNCTION=0
ENV NUMPY_WARN_UNALIGNED_ACCESS=0

# Run the application
CMD ["python", "server.py"]
