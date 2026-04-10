# frontend.py
import streamlit as st
import requests
from PIL import Image
import numpy as np

st.title("AI Fake Face Detection")

uploaded_file = st.file_uploader("Upload an image", type=["jpg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Analyze"):
        files = {"file": uploaded_file}
        response = requests.post("http://127.0.0.1:5000/predict", files=files)
        data = response.json()

        st.subheader("Result")
        st.write(f"Prediction: {data['result']}")
        st.write(f"Confidence: {data['confidence']}%")

        heatmap = np.array(data['heatmap']).astype(np.uint8)
        st.image(heatmap, caption="Grad-CAM Heatmap", use_column_width=True)