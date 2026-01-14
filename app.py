import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import os

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="AI Health Monitor", page_icon="🫁", layout="centered")

st.title("🫁 AI Audio Health Monitor")
st.write("Upload a 4-second audio clip (WAV, MP3, WEBM) to detect patterns.")

# 2. LOAD ASSETS (Cached for speed)
@st.cache_resource
def load_assets():
    # Load Model
    model = tf.keras.models.load_model('audio_cnn_final.keras')
    
    # Load Normalization Files
    norm_mean = np.load("norm_mean.npy")
    norm_std = np.load("norm_std.npy")
    
    return model, norm_mean, norm_std

try:
    with st.spinner("Loading AI Brain..."):
        model, norm_mean, norm_std = load_assets()
except Exception as e:
    st.error(f"⚠️ System Error: {e}")
    st.stop()

# 3. PREPROCESSING FUNCTION
def preprocess_audio(file_path, mean, std):
    SAMPLE_RATE = 22050
    DURATION = 4
    SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
    N_MELS = 128

    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

        # Pad or Cut to 4 seconds
        if len(y) > SAMPLES_PER_TRACK:
            y = y[:SAMPLES_PER_TRACK]
        else:
            padding = int(SAMPLES_PER_TRACK - len(y))
            y = np.pad(y, (0, padding), 'constant')

        # Convert to Spectrogram
        spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
        spec_db = librosa.power_to_db(spec, ref=np.max)

        # --- APPLY NORMALIZATION ---
        spec_db = (spec_db - mean) / std
        # ---------------------------

        # Reshape for Model: (1, 128, Time, 1)
        spec_db = spec_db[np.newaxis, ..., np.newaxis]
        return spec_db
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None

# 4. USER INTERFACE
uploaded_file = st.file_uploader("Choose a file...", type=["wav", "mp3", "ogg", "webm"])

if uploaded_file is not None:
    # Save temp file
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.audio("temp_audio.wav")

    if st.button("🔍 Analyze Audio", type="primary"):
        with st.spinner("Analyzing sound patterns..."):
            
            # A. Preprocess
            processed_data = preprocess_audio("temp_audio.wav", norm_mean, norm_std)
            
            if processed_data is not None:
                # B. Predict
                prediction = model.predict(processed_data)
                confidence = np.max(prediction) * 100
                class_index = np.argmax(prediction)

                # C. Result Mapping
                # NOTE: Ensure this order matches your training folders!
                # Usually: 0=Cough, 1=Heavy, 2=Noise (Alphabetical or Data Order)
                classes = ['Cough', 'Heavy Breathing/Respiratory', 'Background Noise', 'Normal']
                
                if class_index < len(classes):
                    result = classes[class_index]
                else:
                    result = "Unknown"

                # D. Display Output
                st.divider()
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(label="Prediction", value=result)
                with col2:
                    st.metric(label="Confidence", value=f"{confidence:.1f}%")

                st.progress(int(confidence))

                # E. Smart Feedback
                if result == "Cough":
                    st.warning("⚠️ **Cough Pattern Detected.** If this persists, consider checking with a doctor.")
                elif result == "Heavy Breathing/Respiratory":
                    st.warning("⚠️ **Respiratory Strain Detected.** This sounds like heavy breathing or wheezing.")
                elif result == "Background Noise":
                    st.info("🔊 **Noise Detected.** This doesn't sound like a clear human respiratory sound.")
                elif result == "Normal":
                    st.success("✅ **Normal Pattern.** This sounds like healthy breathing.")