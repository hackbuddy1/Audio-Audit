---
title: Audio Audit
emoji: 🫁
colorFrom: red
colorTo: red
sdk: docker
app_port: 8501
tags:
  - streamlit
  - audio-classification
  - deep-learning
  - cnn
pinned: false
short_description: AI-based respiratory audio analysis using deep learning
---

# Audio Audit

Audio-Audit is a learning-focused machine learning project that analyzes short respiratory audio clips and classifies them into meaningful categories such as **Cough**, **Heavy Breathing**, **Normal Breathing**, and **Background Noise**.

This project was built as part of my journey from understanding the basics of machine learning to developing and deploying a complete, end-to-end system.

---

## 🔍 What This Project Does

- Accepts short (4–5 second) audio recordings
- Converts audio into **Mel Spectrograms**
- Uses a **CNN-based deep learning model** for classification
- Applies **confidence-based filtering** to avoid uncertain predictions
- Returns **"Unclear / Unknown"** when confidence is low
- Provides a simple **Streamlit web interface** for real-world testing

---

## 🧠 Model & Approach

- Audio preprocessing using **Librosa**
- Fixed-length normalization (4 seconds)
- Feature extraction via **Mel Spectrograms**
- CNN trained on real-world respiratory and noise samples
- Confidence thresholding to reduce false positives

> This system is **not a medical diagnostic tool** and is intended purely for learning and experimentation.

---

## 🧪 Observations & Limitations

- Sharp transient sounds (finger snaps, clicks) can sometimes resemble cough patterns
- Real-world audio is highly noisy and unpredictable
- Model performance is limited by dataset size and diversity

These observations helped me better understand the challenges of deploying ML systems outside controlled environments.

---

## 🖥️ Tech Stack

- Python
- TensorFlow / Keras
- Librosa
- NumPy
- Streamlit
- HuggingFace Spaces
- Docker

---

## 🚀 Deployment

The application is deployed using **HuggingFace Spaces** and runs inside a Docker container.

Users can:
- Upload an audio file
- Record audio directly
- View predictions with confidence scores

---

## 📌 Learning Outcome

This project represents my transition from learning ML concepts to:
- Working with real-world data
- Performing basic audio EDA
- Training and evaluating neural networks
- Deploying an interactive ML application

I plan to continue improving this system with more diverse data and stronger robustness.

---

## ⚠️ Disclaimer

This project is **not intended for medical use**.  
Results should not be interpreted as health advice.

---

**Built with curiosity, mistakes, and learning.**
