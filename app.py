import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import os
import datetime
import uuid
from huggingface_hub import HfApi

# 1. PAGE SETUP
st.set_page_config(page_title="AI Health Monitor", page_icon="🫁")
st.title("🫁 AI Audio Health Monitor")
st.write("Upload a 4-second audio clip to detect patterns.")

# 2. SETUP SAVING SYSTEM (Tijori Connection)
# Hum secret pocket se chaabi nikaal rahe hain
HF_TOKEN = os.getenv("HF_TOKEN")
# YAHAN APNA DATASET NAAM LIKHO (username/dataset_name)
DATASET_REPO_ID = "lakshayupadhyay/collected_audio_data" 

def save_to_cloud(file_path, predicted_label):
    # Debugging: Check if token exists
    if not HF_TOKEN:
        st.error("❌ Error: HF_TOKEN not found in Secrets. Please add it in Settings.")
        return False
    
    if HF_TOKEN and DATASET_REPO_ID:
        try:
            api = HfApi(token=HF_TOKEN)
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            unique_id = str(uuid.uuid4())[:8]
            cloud_filename = f"{predicted_label}/{timestamp}_{unique_id}.wav"
            
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=cloud_filename,
                repo_id=DATASET_REPO_ID,
                repo_type="dataset"
            )
            return True
        except Exception as e:
            st.error(f"❌ Save Failed: {e}")  # Ye humein asli reason batayega
            return False
    return False

# 3. LOAD MODEL
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('audio_cnn_final.keras')
    norm_mean = np.load("norm_mean.npy")
    norm_std = np.load("norm_std.npy")
    return model, norm_mean, norm_std

try:
    with st.spinner("Loading AI Brain..."):
        model, norm_mean, norm_std = load_assets()
except:
    st.error("⚠️ Model files missing.")
    st.stop()

# 4. PREPROCESSING
def preprocess_audio(file_path, mean, std):
    SAMPLE_RATE = 22050
    DURATION = 4
    SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
    N_MELS = 128
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        if len(y) > SAMPLES_PER_TRACK: y = y[:SAMPLES_PER_TRACK]
        else: y = np.pad(y, (0, int(SAMPLES_PER_TRACK - len(y))), 'constant')
        spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
        spec_db = librosa.power_to_db(spec, ref=np.max)
        spec_db = (spec_db - mean) / std
        return spec_db[np.newaxis, ..., np.newaxis]
    except: return None

# 5. UI INTERFACE
uploaded_file = st.file_uploader("Choose a file...", type=["wav", "mp3", "ogg", "webm"])

if uploaded_file is not None:
    # Save local temp copy
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.audio("temp_audio.wav")

    if st.button("🔍 Analyze Audio", type="primary"):
        with st.spinner("Analyzing..."):
            processed_data = preprocess_audio("temp_audio.wav", norm_mean, norm_std)
            
            if processed_data is not None:
                prediction = model.predict(processed_data)
                confidence = np.max(prediction) * 100
                class_index = np.argmax(prediction)
                
                classes = ['Cough', 'Heavy Breathing', 'Background Noise', 'Normal']
                result = classes[class_index] if class_index < len(classes) else "Unknown"

                # SHOW RESULT
                st.divider()
                c1, c2 = st.columns(2)
                c1.metric("Result", result)
                c2.metric("Confidence", f"{confidence:.1f}%")
                st.progress(int(confidence))

                if result == "Cough": st.warning("⚠️ Cough Pattern Detected.")
                elif result == "Heavy Breathing": st.warning("⚠️ Respiratory Strain Detected.")
                elif result == "Normal": st.success("✅ Sounds Healthy.")
                else: st.info("🔊 Just Background Noise.")

                # --- SAVE DATA (THE FLYWHEEL) ---
                # Hum result aane ke baad file save kar rahe hain
                with st.spinner("Saving data securely..."):
                    saved = save_to_cloud("temp_audio.wav", result)
                    if saved:
                        st.toast("Data contributed anonymously! Thank you. ☁️", icon="✅")