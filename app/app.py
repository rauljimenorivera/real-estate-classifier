"""Streamlit front-end for image inference."""

from __future__ import annotations

import requests
import streamlit as st

st.set_page_config(page_title="Real Estate Classifier", layout="centered")
st.title("Real Estate Image Classifier")
st.caption("Upload an image and classify room/scene type.")

api_url = st.text_input("FastAPI URL", value="http://127.0.0.1:8000/predict")
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Input image", use_container_width=True)

if st.button("Predict", type="primary"):
    if uploaded_file is None:
        st.warning("Please upload an image first.")
    else:
        with st.spinner("Calling inference API..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            try:
                response = requests.post(api_url, files=files, timeout=30)
                response.raise_for_status()
            except requests.RequestException as exc:
                st.error(f"Error calling API: {exc}")
            else:
                payload = response.json()
                st.subheader("Predictions")
                for pred in payload["predictions"]:
                    st.write(f"- {pred['class_name']}: {pred['probability']:.4f}")


