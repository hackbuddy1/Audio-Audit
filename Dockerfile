FROM python:3.9-slim

WORKDIR /app

# 1. Install ESSENTIALS Only
# Removed 'software-properties-common' and 'git' to stop errors
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 2. Copy Files
COPY . .

# 3. Install Python Libraries
RUN pip3 install --no-cache-dir -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# 4. Run App
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]