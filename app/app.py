"""Streamlit front-end for image inference."""

from __future__ import annotations

import requests
import streamlit as st

st.set_page_config(page_title="Real Estate Classifier", layout="centered")
st.title("Real Estate Image Classifier")
st.caption("Upload an image and classify room/scene type.")

api_base_url = st.text_input("FastAPI base URL", value="http://127.0.0.1:8000")
predict_url = f"{api_base_url.rstrip('/')}/predict"
load_model_url = f"{api_base_url.rstrip('/')}/load-model"
models_url = f"{api_base_url.rstrip('/')}/models"

st.subheader("Model selection")
model_source = st.radio("Source", ["W&B artifact", "Local checkpoint"], horizontal=True)
wandb_entity = st.text_input("W&B entity", value="202529987-universidad-pontificia-comillas")
wandb_project = st.text_input("W&B project", value="real-estate-classifier")
artifact_ref = st.text_input(
    "W&B artifact reference",
    value="",
    placeholder="entity/project/best-model:v12",
    disabled=model_source != "W&B artifact",
)
model_path = st.text_input(
    "Local model path",
    value="artifacts/best_model.pth",
    disabled=model_source != "Local checkpoint",
)

if "wandb_model_options" not in st.session_state:
    st.session_state["wandb_model_options"] = []
if "wandb_models_loaded" not in st.session_state:
    st.session_state["wandb_models_loaded"] = False

col_refresh, col_load = st.columns([1, 1])
with col_refresh:
    if st.button("Refresh W&B models", disabled=model_source != "W&B artifact"):
        params = {}
        if wandb_entity.strip():
            params["entity"] = wandb_entity.strip()
        if wandb_project.strip():
            params["project"] = wandb_project.strip()
        with st.spinner("Loading model list from W&B..."):
            try:
                response = requests.get(models_url, params=params, timeout=60)
                response.raise_for_status()
                payload = response.json()
                st.session_state["wandb_model_options"] = payload.get("models", [])
                st.session_state["wandb_models_loaded"] = True
            except requests.RequestException as exc:
                st.error(f"Error fetching W&B models: {exc}")
                st.session_state["wandb_models_loaded"] = False

selected_artifact = None
if model_source == "W&B artifact":
    if st.session_state["wandb_model_options"]:
        selected_artifact = st.selectbox(
            "Available W&B models",
            options=st.session_state["wandb_model_options"],
            index=0,
        )
    elif st.session_state["wandb_models_loaded"]:
        st.warning("No model artifacts found for that entity/project.")
    else:
        st.info("Click 'Refresh W&B models' to load and show the list here.")

if model_source == "W&B artifact" and selected_artifact:
    artifact_ref = selected_artifact

with col_load:
    load_clicked = st.button("Load selected model")

if load_clicked:
    if model_source == "W&B artifact":
        payload = {"artifact_ref": artifact_ref.strip() or None}
        if not payload["artifact_ref"]:
            st.warning("Write a valid W&B artifact reference first.")
            payload = None
    else:
        payload = {"model_path": model_path.strip() or None}
        if not payload["model_path"]:
            st.warning("Write a valid local checkpoint path first.")
            payload = None

    if payload is not None:
        with st.spinner("Loading model..."):
            try:
                response = requests.post(load_model_url, json=payload, timeout=90)
                response.raise_for_status()
            except requests.RequestException as exc:
                st.error(f"Error loading model: {exc}")
            else:
                info = response.json()
                st.success(f"Model loaded: {info.get('model_source', 'unknown')}")

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
                response = requests.post(predict_url, files=files, timeout=30)
                response.raise_for_status()
            except requests.RequestException as exc:
                st.error(f"Error calling API: {exc}")
            else:
                payload = response.json()
                st.subheader("Predictions")
                for pred in payload["predictions"]:
                    st.write(f"- {pred['class_name']}: {pred['probability']:.4f}")


