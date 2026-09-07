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