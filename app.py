<<<<<<< HEAD
"""
Agricultural Field Boundary (Bund) Extraction — Streamlit App
================================================================
Upload a satellite image (jpg/png/tif) and get a predicted field
boundary (bund) mask using a U-Net (ResNet34 encoder) model.

Deploy on Hugging Face Spaces (SDK: streamlit). See README.md for
deployment steps.
"""

import io
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import segmentation_models_pytorch as smp
from skimage.morphology import skeletonize

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Bund Extraction — Field Boundary Detection",
    page_icon="🌾",
    layout="wide",
)

from huggingface_hub import hf_hub_download

# ----------------------------------------------------------------------
# IMPORTANT: change this to your own HF model repo once you create it
# (see README.md for how to upload your .pth file there)
# ----------------------------------------------------------------------
HF_MODEL_REPO = "Han-si1/bund-extraction-weights"
HF_MODEL_FILENAME = "best_bund_model.pth"
IMG_SIZE = 256  # must match the size used during training


# ----------------------------------------------------------------------
# Model loading (cached so it only downloads/loads once per session)
# ----------------------------------------------------------------------
@st.cache_resource
def load_model():
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,  # weights come from our checkpoint, not ImageNet, at inference time
        in_channels=3,
        classes=1,
        activation=None,
    )
    weights_path = hf_hub_download(repo_id=HF_MODEL_REPO, filename=HF_MODEL_FILENAME)
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def read_image_any_format(uploaded_file):
    """Read jpg/png/tif into a normalized (H, W, 3) float32 array in [0, 1].
    Falls back gracefully across formats without requiring GDAL for
    ordinary jpg/png uploads."""
    name = uploaded_file.name.lower()
    raw_bytes = uploaded_file.read()

    if name.endswith((".tif", ".tiff")):
        import rasterio
        from rasterio.io import MemoryFile

        with MemoryFile(raw_bytes) as memfile:
            with memfile.open() as src:
                arr = src.read()  # (bands, H, W)
        arr = np.transpose(arr, (1, 2, 0)).astype(np.float32)
        arr = arr[:, :, :3] if arr.shape[2] >= 3 else np.repeat(arr, 3, axis=2)
    else:
        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        arr = np.array(img).astype(np.float32)

    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
    return arr


def preprocess(arr, size=IMG_SIZE):
    """Resize to model input size and convert to a torch tensor."""
    img = Image.fromarray((arr * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR)
    arr_resized = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr_resized.transpose(2, 0, 1)).unsqueeze(0).float()
    return tensor, arr_resized


def run_inference(model, tensor, threshold=0.5):
    with torch.no_grad():
        logits = model(tensor)
        prob = torch.sigmoid(logits)[0, 0].numpy()
    binary_mask = (prob > threshold).astype(np.uint8)
    return prob, binary_mask


def mask_to_png_bytes(mask_uint8_0_255):
    img = Image.fromarray(mask_uint8_0_255)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def overlay_mask_on_image(rgb_float, binary_mask, color=(255, 60, 60), alpha=0.6):
    base = (rgb_float * 255).astype(np.uint8).copy()
    overlay = base.copy()
    overlay[binary_mask.astype(bool)] = color
    blended = (alpha * overlay + (1 - alpha) * base).astype(np.uint8)
    return blended


# ----------------------------------------------------------------------
# Sidebar — project info
# ----------------------------------------------------------------------
with st.sidebar:
    st.title("🌾 About This Project")
    st.markdown(
        """
**Agricultural Field Boundary (Bund) Extraction**

A deep learning pipeline that detects field boundaries ("bunds")
from satellite imagery using semantic segmentation.

**Model:** U-Net, ResNet34 encoder (ImageNet pretrained)
**Loss:** 0.5 × Dice + 0.5 × BCE
**Training data:** Fields of the World (FTW) dataset, Luxembourg subset

**Validation metrics:**
| Metric | Score |
|---|---|
| IoU | 0.50 |
| F1 | 0.67 |
| Precision | 0.64 |
| Recall | 0.71 |
"""
    )
    st.divider()
    threshold = st.slider("Prediction threshold", 0.1, 0.9, 0.5, 0.05)
    show_skeleton = st.checkbox("Show skeletonized (1px) boundary", value=False)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
st.title("Field Boundary (Bund) Extraction")
st.caption("Upload a satellite image chip to predict field/bund boundaries.")

uploaded_file = st.file_uploader(
    "Upload a satellite image (JPG, PNG, or TIF)",
    type=["jpg", "jpeg", "png", "tif", "tiff"],
)

if uploaded_file is not None:
    try:
        model = load_model()
    except Exception as e:
        st.error(
            f"Could not load the model from Hugging Face repo `{HF_MODEL_REPO}`.\n\n"
            f"Error: {e}\n\n"
            "Check that: (1) HF_MODEL_REPO at the top of app.py matches your actual "
            "HF model repo name, and (2) best_bund_model.pth was uploaded there."
        )
        st.stop()

    with st.spinner("Running inference..."):
        raw_arr = read_image_any_format(uploaded_file)
        tensor, display_arr = preprocess(raw_arr)
        prob_map, binary_mask = run_inference(model, tensor, threshold=threshold)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Input Image")
        st.image(display_arr, use_container_width=True)

    with col2:
        st.subheader("Predicted Bund Mask")
        mask_display = binary_mask * 255
        if show_skeleton:
            mask_display = (skeletonize(binary_mask.astype(bool)).astype(np.uint8)) * 255
        st.image(mask_display, use_container_width=True, clamp=True)

    with col3:
        st.subheader("Overlay")
        overlay_img = overlay_mask_on_image(display_arr, binary_mask)
        st.image(overlay_img, use_container_width=True)

    st.divider()

    # Stats about this prediction
    pred_pixel_frac = float(binary_mask.mean())
    st.write(
        f"**Predicted boundary pixel coverage:** {pred_pixel_frac*100:.2f}% of image "
        f"(threshold = {threshold})"
    )

    # Downloads
    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        st.download_button(
            "Download predicted mask (PNG)",
            data=mask_to_png_bytes(mask_display.astype(np.uint8)),
            file_name="predicted_bund_mask.png",
            mime="image/png",
        )
    with dl_col2:
        prob_png = mask_to_png_bytes((prob_map * 255).astype(np.uint8))
        st.download_button(
            "Download raw probability map (PNG)",
            data=prob_png,
            file_name="prediction_probability.png",
            mime="image/png",
        )

else:
    st.info("👆 Upload a satellite image chip to get started.")
    st.markdown(
        """
**Tips for best results:**
- Works best on Sentinel-2-style RGB imagery of farmland (similar to the training data)
- Images are automatically resized to 256×256 for the model
- For GeoTIFF inputs, only the RGB bands are used
"""
    )
=======
import streamlit as st

st.set_page_config(
    page_title="Bund Extraction — Home",
    page_icon="🌾",
    layout="wide",
)

st.title("🌾 Agricultural Field Boundary (Bund) Extraction")
st.caption("A deep learning system for detecting field boundaries from satellite imagery")

st.markdown(
    """
Agricultural field boundaries — **bunds** — mark the edges between individual
farm plots. They matter for land parcel demarcation, cadastral mapping,
precision agriculture, and soil and water conservation. Mapping them by hand
from field surveys or imagery is slow and doesn't scale to large agricultural
regions.

This project treats bund extraction as a **semantic segmentation** problem:
given a satellite image, predict a pixel-level mask of where the boundaries
are, using a U-Net model trained on real satellite imagery.
"""
)

st.divider()

st.subheader("At a glance")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Validation IoU", "0.50")
col2.metric("F1-score", "0.67")
col3.metric("Training images", "579")
col4.metric("Validation images", "145")

st.divider()

st.subheader("What you can do here")
nav_col1, nav_col2, nav_col3, nav_col4 = st.columns(4)

with nav_col1:
    st.markdown("### Predict")
    st.write("Upload a satellite image and get a predicted boundary mask.")

with nav_col2:
    st.markdown("### Threshold Explorer")
    st.write("See how the prediction changes as the confidence threshold shifts.")

with nav_col3:
    st.markdown("### Skeletonization")
    st.write("View the post-processed, thinned, GIS-ready boundary lines.")

with nav_col4:
    st.markdown("### Model & Methodology")
    st.write("Architecture, training curves, metrics, and known limitations.")

st.info("👈 Use the sidebar to navigate between pages. Start with **Predict**.")

st.divider()
st.caption(
    "Model: U-Net (ResNet34 encoder) · Dataset: Fields of the World (FTW), Luxembourg subset · "
    "Loss: Dice + BCE"
)
>>>>>>> ba25299 (Added new project files)
