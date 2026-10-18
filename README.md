# Traffic Sign Recognition (GTSRB) – CNN + Transfer Learning

Classifies a cropped traffic-sign image into one of 43 GTSRB classes. Learning prototype only – not for real driving use.

## Setup
```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
TensorFlow is pinned to 2.16–2.19: 2.20 crashes on import on macOS when pyarrow (a Streamlit dependency) is installed, and the saved `.keras` model needs Keras 3. Tested with TensorFlow 2.19.1 on Python 3.9.
## Reproduce training
Open `Traffic_Sign_Recognition.ipynb` and run all cells. It downloads GTSRB from the official source, trains a baseline CNN and a
MobileNetV2 transfer model, evaluates on the official test set and writes `artifacts/` (model, meta.json, metrics, plots).
## Run the demo (no retraining needed)
```
streamlit run app.py
```
Upload a PNG/JPG of a single, tightly cropped sign. The app shows the top-5 classes with confidence.
## Layout
- `Traffic_Sign_Recognition.ipynb` – data access, EDA, models, evaluation
- `artifacts/` – `traffic_sign_model.keras`, `.tflite`, `meta.json`, `experiments.csv`, `test_summary.csv`
- `app.py` – Streamlit interface
## Limitations
Trained on German signs in cropped, controlled crops; will not detect signs in full scenes and may fail on blur, darkness, occlusion or non-German designs.
