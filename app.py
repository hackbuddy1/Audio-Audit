import streamlit as st
import numpy as np 
import librosa
import tensorflow as tf 
import os 

st.set_page_config(page_title='AI Health Monitor')
st.title("AI Audio Health Monitor")
st.write("Upload a 4-second audio clip (WAV, MP3, WEBM).")

@st.cache_resource
def load_assets():
    model = tf.keras.model.load_model('audio_cnn_final.keras')
    mean = np.load("norm_mean.npy")
    std = np.load("norm_std.npy")
    return model, mean, std

try:
    with st.spinner("Loading AI Brain..."):
        model, norm_mean, norm_std = load_assets()

except:
    st.error("Model or Normalization not found! Please upload 'audio_cnn_final.keras' to Files.")

def preprocess_audio(file_path):
    SAMPLE_RATE = 22050
    DURATION = 4
    SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
    N_MELS = 128

    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

        if len(y) > SAMPLES_PER_TRACK:
            y = y[:SAMPLES_PER_TRACK]
        else:
            padding = int(SAMPLES_PER_TRACK - len(y))
            y = np.pad(y, (0, padding), 'constant')

        spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
        spec_db = librosa.power_to_db(spec, ref=np.max)
        spec_db = (spec_db - norm_mean) / norm_std

        spec_db = spec_db[np.newaxis, ..., np.newaxis]
        return spec_db
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None

uploaded_file = st.file_uploader("Choose a file...", type=["wav", "mp3", "ogg", "webm"])

if uploaded_file is not None:
    # Save temp file
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.audio("temp_audio.wav")

    if st.button("Analyze Audio"):
        with st.spinner("AI is listening..."):
             processed_data = preprocess_audio("temp_audio.wav", norm_mean, norm_std)

        if processed_data is not None:
               
                prediction = model.predict(processed_data)
                confidence = np.max(prediction) * 100
                class_index = np.argmax(prediction)
            
                classes = ['Cough', 'Heavy Breathing/Respiratory', 'Background Noise', 'Normal']

                if class_index < len(classes):
                    result = classes[class_index]
                else:
                    result = "Unknown"

                st.metric(label="Prediction", value=result)
                st.progress(int(confidence))
                st.caption(f"Confidence: {confidence:.2f}%")

                if result == "Cough":
                    st.warning("Cough detected.")
                elif result == "Heavy Breathing/Respiratory":
                    st.warning("Heavy breathing detected.")
                else:
                    st.success("Sounds like just Noise/Normal.")
            