# Use Python 3.9 (Sabse Stable version AI ke liye)
FROM python:3.9-slim

WORKDIR /app

# 1. Install Audio Drivers (Ye missing tha!)
# libsndfile1 aur ffmpeg zaroori hain audio padhne ke liye
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 2. Copy Files
COPY . .

# 3. Install Python Libraries
# Hum cache clear karte hain taaki fresh install ho
RUN pip3 install --no-cache-dir -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# 4. Run App
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]