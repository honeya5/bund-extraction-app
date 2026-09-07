# Agricultural Field Boundary (Bund) Extraction — Web App

A multi-page Streamlit app for detecting agricultural field boundaries
("bunds") from satellite imagery using a U-Net (ResNet34 encoder) model.
Trained on the Fields of the World (FTW) dataset, Luxembourg subset.

**Validation metrics:** IoU 0.50 · F1 0.67 · Precision 0.64 · Recall 0.71

## Pages

- **🏠 Home** (`app.py`) — welcome dashboard with project overview and key stats
- **🔍 Predict** — upload a satellite image, get a predicted boundary mask,
  overlay, area estimate, and downloads
- **🎚️ Threshold Explorer** — interactively adjust the decision threshold
  on the last prediction, no re-upload needed
- **🧵 Skeletonization** — view the post-processed, thinned boundary lines
  on the last prediction
- **📊 Model & Methodology** — architecture diagrams, training curves,
  metrics table, limitations, and planned extensions

Pages 2 and 3 (Threshold Explorer, Skeletonization) reuse the prediction
from the Predict page via `st.session_state` — you only upload an image once.

## Repository structure

```
.
├── app.py                              # Home page (entry point)
├── utils.py                            # shared model/image logic
├── pages/
│   ├── 1_🔍_Predict.py
│   ├── 2_🎚️_Threshold_Explorer.py
│   ├── 3_🧵_Skeletonization.py
│   └── 4_📊_Model_and_Methodology.py
├── assets/
│   ├── architecture_diagram.png
│   ├── unet_architecture.png
│   └── training_curves.png
├── requirements.txt
└── README.md
```

Streamlit automatically turns any `.py` files inside a `pages/` folder
into extra pages with sidebar navigation — no extra configuration needed.
The emoji + number prefix in each filename controls the icon and sidebar order.

## Model weights

Weights are NOT stored in this repo. They live in a separate free
Hugging Face **model** repo and are downloaded automatically by
`utils.py` the first time the app needs them (see `HF_MODEL_REPO` at
the top of `utils.py`).

## Deployment (Streamlit Community Cloud + Hugging Face Model Hub)

### Step 1 — Model weights on Hugging Face
Already done if you followed the earlier setup: weights live at your
`HF_MODEL_REPO` (e.g. `your-username/bund-extraction-weights`).

### Step 2 — Push this code to GitHub
Upload all files above (keeping the folder structure — `pages/` and
`assets/` must stay as folders) to your GitHub repo via "Add file" →
"Upload files" (drag whole folders, or upload file by file with the
matching path typed in).

### Step 3 — Deploy / redeploy on Streamlit Community Cloud
If you already deployed once, Streamlit Cloud auto-redeploys on every
GitHub commit — no changes needed to your existing app settings, since
the main file path is still `app.py`.

## Extending this project next semester

- **Multi-region generalization** — train/evaluate on additional FTW
  country subsets so the model isn't Luxembourg-specific
- **Vector output** — add polygon/line vectorization (e.g. via
  `rasterio.features.shapes`) and a GeoJSON export button
- **History page** — add a Firebase (Firestore + Storage) backend to
  save past predictions and show them on a new "History" page
- **Sample gallery** — bundle a few known validation chips so reviewers
  can test the app without needing their own image
- **Live evaluation mode** — let users upload a ground-truth mask
  alongside their image to get live IoU/F1 for that specific image