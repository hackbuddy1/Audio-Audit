import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import os
import datetime
import uuid
from huggingface_hub import HfApi

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="RespireAI - Health Monitor",
    page_icon="🫁",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/lungs.png", width=80)
    st.title("RespireAI")
    st.markdown("---")
    st.write("**How to use:**")
    st.write("1. Choose 'Upload' or 'Record'.")
    st.write("2. Provide a 4-5 second cough/breath sound.")
    st.write("3. Get AI Analysis instantly.")
    st.markdown("---")
    st.info("🔒 Data contributed anonymously for research.")
    st.caption("v2.0.0 | Live Recording Added")

# --- MAIN HEADER ---
st.title("🫁 AI Personal Audio Health Monitor")
st.write("Detect respiratory patterns using Artificial Intelligence.")

# 2. SETUP SAVING SYSTEM
HF_TOKEN = os.getenv("HF_TOKEN")
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
        except:
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
    st.error("⚠️ System Error: Model files missing.")
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

# 5. USER INTERFACE (UPDATED WITH TABS)
# Humne yahan Tabs bana diye hain
tab1, tab2 = st.tabs(["📁 Upload File", "🎙️ Record Voice"])

audio_file = None # Final file jo hum process karenge

with tab1:
    uploaded_file = st.file_uploader("Upload Audio (WAV, MP3)", type=["wav", "mp3", "ogg", "webm"])
    if uploaded_file:
        audio_file = uploaded_file

with tab2:
    # Ye Naya Feature hai (Streamlit Audio Input)
    recorded_audio = st.audio_input("Click to Record (4-5 seconds)")
    if recorded_audio:
        audio_file = recorded_audio

# --- ANALYSIS LOGIC ---
if audio_file is not None:
    # Save temp file
    with open("temp_audio.wav", "wb") as f:
        f.write(audio_file.getbuffer())
    
    # Show Audio Player
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

                # SHOW RESULT
                st.divider()
                st.subheader("Analysis Result:")
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    if result == "Cough":
                        st.error(f"⚠️ **{result} Detected**")
                    elif result == "Heavy Breathing":
                        st.warning(f"⚠️ **{result} Detected**")
                    elif result == "Normal":
                        st.success(f"✅ **{result} Pattern**")
                    else:
                        st.info(f"🔊 **{result}**")

                with col2:
                    st.metric("Confidence", f"{confidence:.1f}%")

                st.progress(int(confidence))

                # SAVE DATA
                save_to_cloud("temp_audio.wav", result)
                st.toast("Data contributed for research.", icon="☁️")