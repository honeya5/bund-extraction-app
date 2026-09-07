import numpy as np
import streamlit as st
from skimage.morphology import skeletonize

from utils import mask_to_png_bytes, ensure_prediction_in_session

st.set_page_config(page_title="Skeletonization — Bund Extraction", page_icon="🧵", layout="wide")

st.title("🧵 Skeletonization (Post-Processing)")
st.caption(
    "Raw predicted masks are a few pixels thick. Skeletonization thins them to a clean "
    "1-pixel-wide centerline while preserving the boundary network's topology (junctions, "
    "loops) — the necessary step before converting the mask into GIS-ready vector lines."
)

if ensure_prediction_in_session():
    binary_mask = st.session_state["binary_mask"]

    skeleton = skeletonize(binary_mask.astype(bool)).astype(np.uint8)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Raw Predicted Mask")
        st.image(binary_mask * 255, use_container_width=True, clamp=True)
        st.caption(f"Boundary pixel coverage: {binary_mask.mean()*100:.2f}%")

    with col2:
        st.subheader("Skeletonized (1px centerline)")
        st.image(skeleton * 255, use_container_width=True, clamp=True)
        st.caption(f"Boundary pixel coverage: {skeleton.mean()*100:.2f}%")

    st.divider()
    st.markdown(
        """
**Why this matters:** a thin, 1-pixel-wide mask is what you'd feed into a
vectorization step (e.g. `rasterio.features.shapes` or a skeleton-to-graph
algorithm) to produce actual line/polygon geometry — usable directly in a
GIS tool like QGIS, rather than just a picture of where boundaries are.
This vectorization step is a **planned extension** for next semester (see
the Model & Methodology page).
"""
    )

    st.download_button(
        "Download skeletonized mask (PNG)",
        data=mask_to_png_bytes((skeleton * 255).astype(np.uint8)),
        file_name="skeletonized_bund_mask.png",
        mime="image/png",
    )
