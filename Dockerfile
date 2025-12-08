#official Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies
# gcc/g++ required for webrtcvad
# libsndfile1 required for soundfile
# ffmpeg optional but recommended for audio processing
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage cache
COPY requirements.txt .

# Install uv
RUN pip install uv

# Install Python dependencies using uv
RUN uv pip install --system --no-cache -r requirements.txt

# Copy application code
COPY . .

# Create data directory
RUN mkdir -p data

# Expose API port
EXPOSE 8000

# Run the API server
CMD ["python", "main.py", "serve", "--host", "0.0.0.0", "--port", "8000"]
