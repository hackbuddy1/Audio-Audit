import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import os
import datetime
import uuid
from huggingface_hub import HfApi


st.set_page_config(
    page_title="Audio-Audit",
    layout="centered",
    initial_sidebar_state="expanded"
)


CONFIDENCE_THRESHOLD = 75 


with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/lungs.png", width=80)
    st.title("Audio-Audit")
    st.markdown("---")
    st.write("**How to use:**")
    st.write("1. Choose 'Upload' or 'Record'.")
    st.write("2. Provide a 4-5 second cough/breath sound.")
    st.write("3. Get AI Analysis instantly.")
    st.markdown("---")
    st.info("Data contributed anonymously for research.")
    st.caption("v2.1.0 | High Precision Mode")


st.title("Personal Audio Health Monitor")
st.write("Detect respiratory patterns using Artificial Intelligence.")


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


@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('audio_cnn_final.keras')
    norm_mean = np.load("norm_mean.npy")
    norm_std = np.load("norm_std.npy")
    return model, norm_mean, norm_std

try:
    model, norm_mean, norm_std = load_assets()
except:
    st.error("System Error: Model files missing.")
    st.stop()


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


tab1, tab2 = st.tabs(["Upload File", "Record Voice"])
audio_file = None 

with tab1:
    uploaded_file = st.file_uploader("Upload Audio (WAV, MP3)", type=["wav", "mp3", "ogg", "webm"])
    if uploaded_file: audio_file = uploaded_file

with tab2:
    recorded_audio = st.audio_input("Click to Record (4-5 seconds)")
    if recorded_audio: audio_file = recorded_audio


if audio_file is not None:
    with open("temp_audio.wav", "wb") as f:
        f.write(audio_file.getbuffer())
    
    st.audio("temp_audio.wav")

    if st.button("Run Analysis", type="primary"):
        with st.spinner("Analyzing sound waves..."):
            processed_data = preprocess_audio("temp_audio.wav", norm_mean, norm_std)
            
            if processed_data is not None:
                prediction = model.predict(processed_data)
                confidence = np.max(prediction) * 100
                class_index = np.argmax(prediction)
                
                classes = ['Cough', 'Heavy Breathing', 'Background Noise', 'Normal']
                raw_result = classes[class_index] if class_index < len(classes) else "Unknown"

                
                if confidence < CONFIDENCE_THRESHOLD:
                    final_result = "Unclear / Unknown"
                    is_safe = False
                else:
                    final_result = raw_result
                    is_safe = True
                
                
                st.divider()
                st.subheader("Analysis Result:")
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    if not is_safe:
                       
                        st.info(f"**{final_result}**")
                        st.write("The audio is not clear enough. It might be speech, whistling, or random noise.")
                        st.write("Please record a clear Cough or Breath sound.")
                    
                    elif final_result == "Cough":
                        st.error(f"**{final_result} Detected**")
                        st.write("High confidence cough pattern detected.")
                    
                    elif final_result == "Heavy Breathing":
                        st.warning(f"**{final_result} Detected**")
                        st.write("Signs of respiratory strain detected.")
                    
                    elif final_result == "Normal":
                        st.success(f"**{final_result} Pattern**")
                        st.write("Breathing sounds healthy.")
                    
                    else:
                        st.info(f"**{final_result}**")
                        st.write("This appears to be background noise.")

                with col2:
                    st.metric("AI Confidence", f"{confidence:.1f}%")

                st.progress(int(confidence))

                
                save_to_cloud("temp_audio.wav", final_result)
                if is_safe:
                    st.toast("Result verified.")
                else:
                    st.toast("Low confidence sample saved.")