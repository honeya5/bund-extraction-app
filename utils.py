"""
Shared utilities for the Bund Extraction app — model loading, image I/O,
preprocessing, and inference. Imported by app.py (Home) and every page
in pages/, so the model is loaded once and logic isn't duplicated.
"""

import io
import numpy as np
import streamlit as st
import torch
from PIL import Image
import segmentation_models_pytorch as smp
from huggingface_hub import hf_hub_download

# ----------------------------------------------------------------------
# Config — change HF_MODEL_REPO to your own HF model repo
# ----------------------------------------------------------------------
HF_MODEL_REPO = "Han-si1/bund-extraction-weights"
HF_MODEL_FILENAME = "best_bund_model.pth"
IMG_SIZE = 256  # must match the size used during training

# Sentinel-2 resolution used for training data, for area estimation
METERS_PER_PIXEL_AT_TRAINING_RES = 10  # Sentinel-2 native resolution
# Note: since inputs are resized to IMG_SIZE regardless of original
# resolution, this area estimate is an illustrative approximation, not
# a precise georeferenced measurement — that requires the image's real
# ground sampling distance, only available for true GeoTIFF inputs.


@st.cache_resource
def load_model():
    """Load the U-Net model, downloading weights from Hugging Face
    Model Hub the first time (cached for the rest of the session)."""
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
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
    """Read jpg/png/tif into a normalized (H, W, 3) float32 array in [0, 1]."""
    name = uploaded_file.name.lower()
    raw_bytes = uploaded_file.read()

    if name.endswith((".tif", ".tiff")):
        import rasterio
        from rasterio.io import MemoryFile

        with MemoryFile(raw_bytes) as memfile:
            with memfile.open() as src:
                arr = src.read()
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


def mask_to_png_bytes(mask_uint8):
    img = Image.fromarray(mask_uint8)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def overlay_mask_on_image(rgb_float, binary_mask, color=(255, 60, 60), alpha=0.6):
    base = (rgb_float * 255).astype(np.uint8).copy()
    overlay = base.copy()
    overlay[binary_mask.astype(bool)] = color
    blended = (alpha * overlay + (1 - alpha) * base).astype(np.uint8)
    return blended


def estimate_field_area_hectares(binary_mask, meters_per_pixel=METERS_PER_PIXEL_AT_TRAINING_RES):
    """Rough illustrative estimate of open-field (non-boundary) area in
    hectares, assuming the chip covers IMG_SIZE x meters_per_pixel meters
    per side. This is approximate — real area requires the source
    image's true ground sampling distance."""
    non_boundary_pixels = (binary_mask == 0).sum()
    area_m2 = non_boundary_pixels * (meters_per_pixel ** 2)
    return area_m2 / 10000.0  # m^2 -> hectares


def ensure_prediction_in_session():
    """Helper for pages that depend on a prediction already existing.
    Returns True if session_state has a prediction, else shows a
    friendly prompt and returns False."""
    if "prob_map" not in st.session_state:
        st.info(
            "No image has been analyzed yet. Go to the **Predict** page first, "
            "upload a satellite image, then come back here."
        )
        return False
    return True
