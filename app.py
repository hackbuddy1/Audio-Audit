import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import os
import datetime
import uuid
from huggingface_hub import HfApi

# 1. PAGE CONFIGURATION (Professional Look)
st.set_page_config(
    page_title="RespireAI - Health Monitor",
    page_icon="🫁",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- SIDEBAR (About Us & Instructions) ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/lungs.png", width=80)
    st.title("RespireAI")
    st.markdown("---")
    st.write("**How to use:**")
    st.write("1. Record a 4-5 second clip of coughing or breathing.")
    st.write("2. Upload the audio file.")
    st.write("3. Get instant AI analysis.")
    st.markdown("---")
    st.info("🔒 Your data is contributed anonymously to help train better medical AI models.")
    st.markdown("---")
    st.caption("v1.0.0 | Beta Version")

# --- MAIN HEADER ---
st.title("🫁 AI Personal Audio Health Monitor")
st.markdown("### Detect respiratory patterns using Artificial Intelligence.")
st.markdown("*(Upload WAV, MP3, or WEBM files)*")

# 2. SETUP SAVING SYSTEM
HF_TOKEN = os.getenv("HF_TOKEN")
# 👇 YAHAN APNA USERNAME VERIFY KAR LO
DATASET_REPO_ID = "laksh52/collected_audio_data" 

def save_to_cloud(file_path, predicted_label):
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
            return False
    return False

# 3. LOAD ASSETS
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('audio_cnn_final.keras')
    norm_mean = np.load("norm_mean.npy")
    norm_std = np.load("norm_std.npy")
    return model, norm_mean, norm_std

try:
    model, norm_mean, norm_std = load_assets()
except:
    st.error("⚠️ System Maintenance: AI Model is offline.")
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

# 5. USER INTERFACE
uploaded_file = st.file_uploader(" ", label_visibility="collapsed", type=["wav", "mp3", "ogg", "webm"])

if uploaded_file is not None:
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.audio("temp_audio.wav")

    if st.button("🔍 Run Analysis", type="primary"):
        with st.spinner("Analyzing sound waves..."):
            processed_data = preprocess_audio("temp_audio.wav", norm_mean, norm_std)
            
            if processed_data is not None:
                prediction = model.predict(processed_data)
                confidence = np.max(prediction) * 100
                class_index = np.argmax(prediction)
                
                classes = ['Cough', 'Heavy Breathing', 'Background Noise', 'Normal']
                result = classes[class_index] if class_index < len(classes) else "Unknown"

                # SHOW RESULT CARD
                st.divider()
                st.subheader("Analysis Result:")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    if result == "Cough":
                        st.error(f"⚠️ **{result} Pattern Detected**")
                        st.write("The audio shows characteristics of a cough.")
                    elif result == "Heavy Breathing":
                        st.warning(f"⚠️ **{result} Detected**")
                        st.write("Signs of respiratory strain or wheezing detected.")
                    elif result == "Normal":
                        st.success(f"✅ **{result} Breathing**")
                        st.write("No abnormal patterns detected.")
                    else:
                        st.info(f"🔊 **{result}**")
                        st.write("This appears to be background noise.")

                with col2:
                    st.metric("AI Confidence", f"{confidence:.1f}%")

                # SAVE DATA
                save_to_cloud("temp_audio.wav", result)
                st.toast("Data saved for research.", icon="☁️")

# --- DISCLAIMER (VERY IMPORTANT) ---
st.markdown("---")
st.warning("⚠️ **Disclaimer:** This AI tool is for educational & research purposes only. It is NOT a medical device and cannot diagnose diseases. Always consult a real doctor for health issues.")