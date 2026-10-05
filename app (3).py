
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from huggingface_hub import hf_hub_download


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="PneumoDetect AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(20, 184, 166, 0.08), transparent 25%),
        radial-gradient(circle at 90% 20%, rgba(59, 130, 246, 0.08), transparent 25%),
        #07111f;
    color: #e5eef7;
}

/* Sidebar */

section[data-testid="stSidebar"] {
    background: #081522;
    border-right: 1px solid rgba(148, 163, 184, 0.15);
}

section[data-testid="stSidebar"] * {
    color: #dce8f2;
}

.sidebar-logo {
    text-align: center;
    padding: 10px 0 25px 0;
}

.sidebar-logo .icon {
    font-size: 42px;
}

.sidebar-logo h2 {
    margin: 5px 0 2px 0;
    color: #5eead4;
    font-size: 22px;
}

.sidebar-logo p {
    color: #8da4b8;
    font-size: 11px;
    letter-spacing: 2px;
}

/* Hero */

.hero {
    padding: 55px 10px 45px 10px;
}

.hero-badge {
    display: inline-block;
    padding: 8px 15px;
    border-radius: 20px;
    background: rgba(20, 184, 166, 0.12);
    border: 1px solid rgba(20, 184, 166, 0.3);
    color: #5eead4;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1.2px;
    margin-bottom: 20px;
}

.hero h1 {
    font-size: 48px;
    line-height: 1.08;
    font-weight: 800;
    margin: 0;
    color: #f8fafc;
}

.hero p {
    max-width: 800px;
    color: #9fb2c5;
    font-size: 16px;
    line-height: 1.8;
    margin-top: 22px;
}

/* Cards */

.card {
    background: rgba(15, 31, 48, 0.82);
    border: 1px solid rgba(148, 163, 184, 0.14);
    border-radius: 18px;
    padding: 24px;
    margin-bottom: 18px;
}

.card h3 {
    color: #f1f5f9;
    margin-top: 0;
}

.card p {
    color: #9fb2c5;
    line-height: 1.7;
}

/* Metrics */

.metric-card {
    background: linear-gradient(
        145deg,
        rgba(15, 35, 53, 0.95),
        rgba(9, 25, 40, 0.95)
    );
    border: 1px solid rgba(94, 234, 212, 0.15);
    border-radius: 16px;
    padding: 22px;
    text-align: center;
}

.metric-label {
    color: #8fa6ba;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.metric-value {
    color: #5eead4;
    font-size: 30px;
    font-weight: 800;
    margin-top: 8px;
}

/* Section title */

.section-title {
    color: #f8fafc;
    font-size: 28px;
    font-weight: 750;
    margin: 30px 0 18px 0;
}

.section-subtitle {
    color: #8fa6ba;
    margin-bottom: 25px;
}

/* Prediction */

.prediction-normal {
    background: rgba(34, 197, 94, 0.10);
    border: 1px solid rgba(34, 197, 94, 0.30);
    border-radius: 18px;
    padding: 28px;
    text-align: center;
}

.prediction-pneumonia {
    background: rgba(239, 68, 68, 0.10);
    border: 1px solid rgba(239, 68, 68, 0.30);
    border-radius: 18px;
    padding: 28px;
    text-align: center;
}

.prediction-title {
    font-size: 30px;
    font-weight: 800;
    margin-bottom: 8px;
}

.prediction-confidence {
    color: #b7c7d6;
    font-size: 15px;
}

/* Workflow */

.workflow-card {
    background: #0c1b2a;
    border: 1px solid rgba(148, 163, 184, 0.12);
    border-radius: 16px;
    padding: 22px;
    height: 100%;
}

.workflow-number {
    color: #5eead4;
    font-weight: 800;
    font-size: 13px;
    letter-spacing: 1px;
}

.workflow-card h3 {
    color: #f8fafc;
    margin: 10px 0;
}

.workflow-card p {
    color: #91a6ba;
    line-height: 1.6;
}

/* Info */

.info-box {
    background: rgba(59, 130, 246, 0.08);
    border: 1px solid rgba(59, 130, 246, 0.20);
    border-radius: 14px;
    padding: 18px;
    color: #b9c9d8;
    line-height: 1.7;
}

/* Disclaimer */

.disclaimer {
    background: rgba(245, 158, 11, 0.08);
    border: 1px solid rgba(245, 158, 11, 0.22);
    border-radius: 14px;
    padding: 18px;
    color: #d6c39b;
    line-height: 1.7;
}

/* Footer */

.footer {
    text-align: center;
    padding: 35px 10px 20px 10px;
    color: #6f8497;
    font-size: 13px;
    border-top: 1px solid rgba(148, 163, 184, 0.10);
    margin-top: 50px;
}

.footer b {
    color: #5eead4;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# MODEL CONFIG
# =========================================================

MODEL_REPO = "Rushikeshmhaske/pneumodetect-densenet121"
MODEL_FILENAME = "pneumonia_densenet121_final.keras"


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_pneumonia_model():

    try:

        model_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=MODEL_FILENAME
        )

        loaded_model = tf.keras.models.load_model(model_path)

        return loaded_model

    except Exception as e:

        st.error("Unable to load the PneumoDetect AI model.")
        st.exception(e)
        st.stop()


model = load_pneumonia_model()


# =========================================================
# GRAD-CAM FUNCTION
# =========================================================

def make_gradcam_heatmap(img_array):

    try:

        base_model = model.get_layer("densenet121")

        global_pool = model.get_layer(
            "global_average_pooling2d_2"
        )

        dense_256 = model.get_layer("dense_4")

        batch_norm = model.get_layer(
            "batch_normalization_8"
        )

        dense_128 = model.get_layer("dense_5")

        final_dense = model.get_layer("dense_6")

        last_conv = base_model.get_layer(
            "conv5_block16_2_conv"
        )

        grad_model = tf.keras.Model(
            inputs=base_model.input,
            outputs=[
                last_conv.output,
                base_model.output
            ]
        )

        with tf.GradientTape() as tape:

            conv_outputs, features = grad_model(
                img_array,
                training=False
            )

            x = global_pool(features)

            x = dense_256(x)

            x = batch_norm(
                x,
                training=False
            )

            x = dense_128(x)

            x = final_dense(x)

            prediction = x[:, 0]

        grads = tape.gradient(
            prediction,
            conv_outputs
        )

        pooled_grads = tf.reduce_mean(
            grads,
            axis=(1, 2)
        )

        conv_outputs = conv_outputs[0]
        pooled_grads = pooled_grads[0]

        heatmap = tf.reduce_sum(
            conv_outputs * pooled_grads,
            axis=-1
        )

        heatmap = tf.maximum(
            heatmap,
            0
        )

        max_heatmap = tf.reduce_max(
            heatmap
        )

        if max_heatmap > 0:

            heatmap /= max_heatmap

        return (
            heatmap.numpy(),
            float(prediction.numpy()[0])
        )

    except Exception as e:

        st.warning(
            "Grad-CAM could not be generated for this image."
        )

        return None, None


# =========================================================
# IMAGE PREPROCESSING
# =========================================================

def preprocess_image(image):

    image = image.convert("RGB")

    image = image.resize(
        (224, 224)
    )

    image_array = np.array(
        image
    ).astype("float32")

    image_array = image_array / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# =========================================================
# CREATE GRAD-CAM OVERLAY
# =========================================================

def create_gradcam_overlay(
    original_image,
    heatmap
):

    original = np.array(
        original_image.convert("RGB")
    )

    original = cv2.resize(
        original,
        (224, 224)
    )

    heatmap = cv2.resize(
        heatmap,
        (224, 224)
    )

    heatmap = np.uint8(
        255 * heatmap
    )

    heatmap_color = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    heatmap_color = cv2.cvtColor(
        heatmap_color,
        cv2.COLOR_BGR2RGB
    )

    overlay = cv2.addWeighted(
        original,
        0.60,
        heatmap_color,
        0.40,
        0
    )

    return overlay


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("""
    <div class="sidebar-logo">

        <div class="icon">🫁</div>

        <h2>PneumoDetect AI</h2>

        <p>CHEST X-RAY INTELLIGENCE</p>

    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "🩻 Pneumonia Detection",
            "📊 Model Performance",
            "🔬 Explainable AI",
            "ℹ️ About"
        ]
    )

    st.markdown("---")

    st.markdown("""
    <div class="card">

        <h3>AI SYSTEM</h3>

        <p>
        <b>DenseNet121</b><br>
        Transfer Learning<br>
        Fine-Tuned Model
        </p>

        <h3>CLASSES</h3>

        <p>
        • Normal<br>
        • Pneumonia
        </p>

    </div>
    """, unsafe_allow_html=True)


# =========================================================
# DASHBOARD
# =========================================================

if page == "🏠 Dashboard":

    st.markdown("""
    <div class="hero">

        <div class="hero-badge">
            ● AI-POWERED HEALTHCARE SCREENING
        </div>

        <h1>
            Intelligent Pneumonia<br>
            Detection from Chest X-Rays
        </h1>

        <p>
            PneumoDetect AI uses a fine-tuned DenseNet121 deep
            learning model to analyze chest X-ray images and
            provide an AI-assisted pneumonia screening result
            with explainable Grad-CAM visualization.
        </p>

    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Model Performance</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">Evaluated on an untouched test dataset.</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown("""
        <div class="metric-card">

            <div class="metric-label">
                Test Accuracy
            </div>

            <div class="metric-value">
                89.10%
            </div>

        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="metric-card">

            <div class="metric-label">
                ROC-AUC
            </div>

            <div class="metric-value">
                94.78%
            </div>

        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="metric-card">

            <div class="metric-label">
                Pneumonia Recall
            </div>

            <div class="metric-value">
                90.51%
            </div>

        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown("""
        <div class="metric-card">

            <div class="metric-label">
                Pneumonia F1
            </div>

            <div class="metric-value">
                91.21%
            </div>

        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">How It Works</div>',
        unsafe_allow_html=True
    )

    w1, w2, w3 = st.columns(3)

    with w1:

        st.markdown("""
        <div class="workflow-card">

            <div class="workflow-number">
                01 · UPLOAD
            </div>

            <h3>Upload X-Ray</h3>

            <p>
                Upload a chest X-ray image or capture
                an image using your camera.
            </p>

        </div>
        """, unsafe_allow_html=True)

    with w2:

        st.markdown("""
        <div class="workflow-card">

            <div class="workflow-number">
                02 · ANALYZE
            </div>

            <h3>AI Analysis</h3>

            <p>
                DenseNet121 processes the X-ray and
                predicts Normal or Pneumonia.
            </p>

        </div>
        """, unsafe_allow_html=True)

    with w3:

        st.markdown("""
        <div class="workflow-card">

            <div class="workflow-number">
                03 · EXPLAIN
            </div>

            <h3>Grad-CAM</h3>

            <p>
                Explainable AI highlights regions that
                influenced the model prediction.
            </p>

        </div>
        """, unsafe_allow_html=True)


# =========================================================
# PNEUMONIA DETECTION
# =========================================================

elif page == "🩻 Pneumonia Detection":

    st.markdown(
        '<div class="section-title">🩻 Pneumonia Detection</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">Upload a chest X-ray image for AI-assisted screening.</div>',
        unsafe_allow_html=True
    )

    upload_col, camera_col = st.columns(2)

    with upload_col:

        uploaded_file = st.file_uploader(
            "Upload Chest X-Ray",
            type=[
                "jpg",
                "jpeg",
                "png"
            ]
        )

    with camera_col:

        camera_file = st.camera_input(
            "Capture X-Ray Image"
        )

    image_file = (
        uploaded_file
        if uploaded_file is not None
        else camera_file
    )

    if image_file is not None:

        image = Image.open(
            image_file
        ).convert("RGB")

        st.markdown(
            '<div class="section-title">Input Image</div>',
            unsafe_allow_html=True
        )

        image_col1, image_col2 = st.columns(
            [1, 1]
        )

        with image_col1:

            st.image(
                image,
                caption="Uploaded Chest X-Ray",
                use_container_width=True
            )

        with image_col2:

            st.markdown("""
            <div class="info-box">

            <b>Image Processing</b>

            <br><br>

            • RGB conversion<br>
            • Resize to 224 × 224<br>
            • Pixel normalization<br>
            • DenseNet121 inference<br>
            • Grad-CAM explainability

            </div>
            """, unsafe_allow_html=True)

        analyze = st.button(
            "🔍 Analyze X-Ray",
            type="primary",
            use_container_width=True
        )

        if analyze:

            with st.spinner(
                "Analyzing chest X-ray..."
            ):

                img_array = preprocess_image(
                    image
                )

                prediction = model.predict(
                    img_array,
                    verbose=0
                )[0][0]

                pneumonia_probability = float(
                    prediction
                )

                normal_probability = (
                    1 - pneumonia_probability
                )

                if pneumonia_probability >= 0.5:

                    predicted_class = "PNEUMONIA"

                    confidence = (
                        pneumonia_probability * 100
                    )

                    st.markdown(f"""
                    <div class="prediction-pneumonia">

                        <div class="prediction-title">
                            🫁 Pneumonia Detected
                        </div>

                        <div class="prediction-confidence">
                            Model Confidence:
                            <b>{confidence:.2f}%</b>
                        </div>

                    </div>
                    """, unsafe_allow_html=True)

                else:

                    predicted_class = "NORMAL"

                    confidence = (
                        normal_probability * 100
                    )

                    st.markdown(f"""
                    <div class="prediction-normal">

                        <div class="prediction-title">
                            ✓ Normal
                        </div>

                        <div class="prediction-confidence">
                            Model Confidence:
                            <b>{confidence:.2f}%</b>
                        </div>

                    </div>
                    """, unsafe_allow_html=True)

                st.markdown(
                    '<div class="section-title">Prediction Details</div>',
                    unsafe_allow_html=True
                )

                p1, p2 = st.columns(2)

                with p1:

                    st.metric(
                        "Normal Probability",
                        f"{normal_probability * 100:.2f}%"
                    )

                with p2:

                    st.metric(
                        "Pneumonia Probability",
                        f"{pneumonia_probability * 100:.2f}%"
                    )

                # Grad-CAM

                st.markdown(
                    '<div class="section-title">🔬 Explainable AI — Grad-CAM</div>',
                    unsafe_allow_html=True
                )

                heatmap, cam_prediction = make_gradcam_heatmap(
                    img_array
                )

                if heatmap is not None:

                    overlay = create_gradcam_overlay(
                        image,
                        heatmap
                    )

                    g1, g2, g3 = st.columns(3)

                    with g1:

                        st.image(
                            image,
                            caption="Original X-Ray",
                            use_container_width=True
                        )

                    with g2:

                        st.image(
                            heatmap,
                            caption="Grad-CAM Heatmap",
                            use_container_width=True,
                            clamp=True
                        )

                    with g3:

                        st.image(
                            overlay,
                            caption="Grad-CAM Overlay",
                            use_container_width=True
                        )

                    st.markdown("""
                    <div class="info-box">

                    <b>How to interpret Grad-CAM</b>

                    <br><br>

                    Brighter regions indicate areas that
                    contributed more strongly to the model's
                    prediction.

                    Grad-CAM provides model explainability
                    and should not be interpreted as a
                    clinical diagnosis.

                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("""
                <br>

                <div class="disclaimer">

                ⚠️ <b>Medical Disclaimer</b><br><br>

                PneumoDetect AI is an academic deep learning
                research project intended for educational and
                demonstration purposes only.

                It is not a medical diagnostic device and
                should not replace professional medical
                evaluation, radiologist interpretation,
                or clinical decision-making.

                </div>
                """, unsafe_allow_html=True)


# =========================================================
# MODEL PERFORMANCE
# =========================================================

elif page == "📊 Model Performance":

    st.markdown(
        '<div class="section-title">📊 Model Performance</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">Final DenseNet121 model evaluated on the untouched test dataset.</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    metrics = [
        ("Accuracy", "89.10%"),
        ("ROC-AUC", "94.78%"),
        ("Precision", "91.93%"),
        ("Recall", "90.51%")
    ]

    for col, (label, value) in zip(
        [c1, c2, c3, c4],
        metrics
    ):

        with col:

            st.markdown(f"""
            <div class="metric-card">

                <div class="metric-label">
                    {label}
                </div>

                <div class="metric-value">
                    {value}
                </div>

            </div>
            """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Classification Report</div>',
        unsafe_allow_html=True
    )

    st.dataframe(
        {
            "Class": [
                "NORMAL",
                "PNEUMONIA"
            ],
            "Precision": [
                0.8458,
                0.9193
            ],
            "Recall": [
                0.8675,
                0.9051
            ],
            "F1-Score": [
                0.8565,
                0.9121
            ],
            "Support": [
                234,
                390
            ]
        },
        use_container_width=True,
        hide_index=True
    )

    st.markdown(
        '<div class="section-title">Model Architecture</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="card">

    <h3>DenseNet121 Transfer Learning</h3>

    <p>
    The system uses DenseNet121 pretrained on ImageNet
    as the feature extraction backbone. The classification
    head was customized for binary pneumonia detection.
    </p>

    <p>
    <b>Input:</b> 224 × 224 × 3 RGB image<br>
    <b>Backbone:</b> DenseNet121<br>
    <b>Feature Extraction:</b> Global Average Pooling<br>
    <b>Classifier:</b> Dense 256 → BatchNorm → Dropout →
    Dense 128 → Dropout → Sigmoid<br>
    <b>Classes:</b> Normal / Pneumonia
    </p>

    </div>
    """, unsafe_allow_html=True)


# =========================================================
# EXPLAINABLE AI
# =========================================================

elif page == "🔬 Explainable AI":

    st.markdown(
        '<div class="section-title">🔬 Explainable AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">Understanding model decisions using Grad-CAM.</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="card">

    <h3>What is Grad-CAM?</h3>

    <p>
    Gradient-weighted Class Activation Mapping (Grad-CAM)
    is an explainability technique used to visualize the
    image regions that contributed to a neural network's
    prediction.
    </p>

    <p>
    In PneumoDetect AI, Grad-CAM is generated from the
    final convolutional layer of DenseNet121.
    </p>

    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Interpretation</div>',
        unsafe_allow_html=True
    )

    e1, e2, e3 = st.columns(3)

    with e1:

        st.markdown("""
        <div class="workflow-card">

        <div class="workflow-number">
        01 · ORIGINAL
        </div>

        <h3>Chest X-Ray</h3>

        <p>
        The original input image provided to the
        deep learning model.
        </p>

        </div>
        """, unsafe_allow_html=True)

    with e2:

        st.markdown("""
        <div class="workflow-card">

        <div class="workflow-number">
        02 · HEATMAP
        </div>

        <h3>Activation Map</h3>

        <p>
        Highlights regions associated with the
        model's learned features.
        </p>

        </div>
        """, unsafe_allow_html=True)

    with e3:

        st.markdown("""
        <div class="workflow-card">

        <div class="workflow-number">
        03 · OVERLAY
        </div>

        <h3>Model Explanation</h3>

        <p>
        Combines the heatmap with the original
        X-ray for easier interpretation.
        </p>

        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <br>

    <div class="disclaimer">

    ⚠️ <b>Important:</b>

    Grad-CAM shows where the model focused when making
    its prediction. It does not prove the presence or
    absence of disease and should not be considered
    a clinical diagnosis.

    </div>
    """, unsafe_allow_html=True)


# =========================================================
# ABOUT
# =========================================================

elif page == "ℹ️ About":

    st.markdown(
        '<div class="section-title">ℹ️ About PneumoDetect AI</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="card">

    <h3>Project Overview</h3>

    <p>
    PneumoDetect AI is a Deep Learning based academic
    research project designed to demonstrate automated
    pneumonia screening from chest X-ray images.
    </p>

    <p>
    The project combines transfer learning using
    DenseNet121 with Grad-CAM explainability and
    Streamlit deployment.
    </p>

    </div>
    """, unsafe_allow_html=True)

    a1, a2 = st.columns(2)

    with a1:

        st.markdown("""
        <div class="card">

        <h3>Technology Stack</h3>

        <p>
        • Python<br>
        • TensorFlow / Keras<br>
        • DenseNet121<br>
        • OpenCV<br>
        • NumPy<br>
        • Pillow<br>
        • Hugging Face Hub<br>
        • Streamlit
        </p>

        </div>
        """, unsafe_allow_html=True)

    with a2:

        st.markdown("""
        <div class="card">

        <h3>AI Pipeline</h3>

        <p>
        Chest X-Ray<br>
        ↓<br>
        Image Preprocessing<br>
        ↓<br>
        DenseNet121 Feature Extraction<br>
        ↓<br>
        Fine-Tuned Classification Head<br>
        ↓<br>
        Normal / Pneumonia<br>
        ↓<br>
        Grad-CAM Explanation
        </p>

        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer">

    ⚠️ <b>Academic Use Only</b><br><br>

    This application is developed for academic,
    educational and research demonstration purposes.
    It is not intended for clinical diagnosis or
    medical decision-making.

    </div>
    """, unsafe_allow_html=True)


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">

    <b>PneumoDetect AI</b>
    · Deep Learning Healthcare Research Project

    <br>

    Built with TensorFlow, DenseNet121, Grad-CAM & Streamlit

    <br><br>

    © 2026 PneumoDetect AI · Academic Project

</div>
""", unsafe_allow_html=True)
