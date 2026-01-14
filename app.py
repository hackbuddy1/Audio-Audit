import streamlit as st
import os

st.title("🕵️‍♂️ Sherlock Holmes Debugger")

# 1. LIST ALL FILES (Humein dekhna hai file ka naam kya hai)
st.subheader("📂 Files on Server:")
try:
    files = os.listdir('.')
    st.write(files)
except Exception as e:
    st.error(f"Cannot list files: {e}")

# 2. CHECK LIBRARIES
st.subheader("📚 Checking Libraries:")
try:
    import tensorflow as tf
    import librosa
    import numpy as np
    st.success(f"TensorFlow Version: {tf.__version__}")
    st.success(f"Librosa Version: {librosa.__version__}")
    st.success(f"Numpy Version: {np.__version__}")
except Exception as e:
    st.error(f"Library missing: {e}")

# 3. TRY LOADING MODEL (With Real Error)
st.subheader("🧠 Trying to Load Model:")
model_name = 'audio_cnn_final.keras' # Yahan wo naam hai jo code dhoondh raha hai

if model_name in files:
    st.write(f"✅ File '{model_name}' found on disk!")
    try:
        model = tf.keras.models.load_model(model_name)
        st.success("🎉 Model Loaded Successfully!")
    except Exception as e:
        st.error(f"❌ File exists, but failed to load. Real Error:")
        st.code(e)
else:
    st.error(f"❌ File '{model_name}' NOT found on disk.")
    st.write("Did you name it 'Audio_cnn_final.keras' (Capital A)? or 'model.keras'?")

# 4. TRY LOADING NUMPY FILES
st.subheader("📊 Trying to Load Normalization Files:")
try:
    mean = np.load("norm_mean.npy")
    std = np.load("norm_std.npy")
    st.success("✅ Normalization files loaded!")
except Exception as e:
    st.error(f"❌ Normalization load failed: {e}")