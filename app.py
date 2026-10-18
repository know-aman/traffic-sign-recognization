import json, numpy as np, streamlit as st
from PIL import Image
import tensorflow as tf

ART = "artifacts"
@st.cache_resource
def load():
    meta = json.load(open(f"{ART}/meta.json"))
    return tf.keras.models.load_model(f"{ART}/traffic_sign_model.keras"), meta

model, meta = load()
S = meta["img_size"]; names = meta["class_names"]

st.title("🚦 Traffic Sign Recognition (GTSRB)")
st.caption("Learning prototype – NOT a safety-certified driving component. Upload a tightly cropped image of a single sign.")
f = st.file_uploader("Upload a traffic sign image", type=["png", "jpg", "jpeg", "ppm"])
if f:
    img = Image.open(f).convert("RGB")
    st.image(img, caption="Input", width=250)
    x = np.asarray(img.resize((S, S), Image.BILINEAR), dtype="float32")[None]
    p = model.predict(x, verbose=0)[0]
    top = p.argsort()[::-1][:5]
    st.subheader(f"Prediction: {names[top[0]]}  ({p[top[0]]:.1%})")
    if p[top[0]] < 0.6:
        st.warning("Low confidence – the image may be blurry, dark, badly cropped or not a GTSRB-style sign.")
    st.bar_chart({names[i]: float(p[i]) for i in top})