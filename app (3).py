

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from huggingface_hub import hf_hub_download


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PneumoDetect AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(13,148,136,0.08), transparent 25%),
        radial-gradient(circle at 90% 20%, rgba(14,116,144,0.07), transparent 25%),
        #f7fafc;
}

/* Sidebar */

section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #062c3d 0%,
        #073b4c 50%,
        #075e68 100%
    );
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

/* Main container */

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

/* Logo */

.logo-box {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 25px;
}

.logo-icon {
    width: 48px;
    height: 48px;
    border-radius: 14px;
    background: linear-gradient(135deg, #14b8a6, #0f766e);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 25px;
    box-shadow: 0 8px 20px rgba(20,184,166,0.25);
}

.logo-title {
    font-size: 20px;
    font-weight: 800;
    letter-spacing: -0.5px;
}

.logo-subtitle {
    font-size: 11px;
    opacity: 0.75;
}

/* Hero */

.hero {
    padding: 40px;
    border-radius: 24px;
    background:
        linear-gradient(
            135deg,
            #073b4c 0%,
            #075e68 55%,
            #0f766e 100%
        );
    color: white;
    box-shadow: 0 18px 45px rgba(7,59,76,0.15);
    margin-bottom: 28px;
}

.hero-badge {
    display: inline-block;
    padding: 7px 13px;
    border-radius: 30px;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.18);
    font-size: 12px;
    font-weight: 600;
    margin-bottom: 16px;
}

.hero h1 {
    font-size: 42px;
    line-height: 1.1;
    margin: 0;
    font-weight: 800;
}

.hero p {
    font-size: 16px;
    opacity: 0.88;
    max-width: 720px;
    line-height: 1.7;
    margin-top: 16px;
}

/* Cards */

.metric-card {
    background: white;
    padding: 22px;
    border-radius: 18px;
    border: 1px solid #e5edf0;
    box-shadow: 0 8px 25px rgba(15,23,42,0.05);
}

.metric-label {
    color: #64748b;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.7px;
}

.metric-value {
    color: #073b4c;
    font-size: 30px;
    font-weight: 800;
    margin-top: 6px;
}

/* Result */

.result-card {
    padding: 28px;
    border-radius: 20px;
    background: white;
    border: 1px solid #e2e8f0;
    box-shadow: 0 10px 30px rgba(15,23,42,0.06);
}

.result-title {
    color: #64748b;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
}

.result-pneumonia {
    color: #dc2626;
    font-size: 34px;
    font-weight: 800;
    margin-top: 5px;
}

.result-normal {
    color: #059669;
    font-size: 34px;
    font-weight: 800;
    margin-top: 5px;
}

/* Section heading */

.section-title {
    color: #073b4c;
    font-size: 25px;
    font-weight: 800;
    margin-top: 30px;
    margin-bottom: 8px;
}

.section-subtitle {
    color: #64748b;
    font-size: 14px;
    margin-bottom: 20px;
}

/* Disclaimer */

.disclaimer {
    padding: 18px 20px;
    border-radius: 15px;
    background: #fff7ed;
    border: 1px solid #fed7aa;
    color: #7c2d12;
    font-size: 13px;
    line-height: 1.6;
}

/* Footer */

.footer {
    margin-top: 50px;
    padding-top: 20px;
    border-top: 1px solid #e2e8f0;
    text-align: center;
    color: #94a3b8;
    font-size: 12px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HUGGING FACE MODEL CONFIGURATION
# ============================================================

MODEL_REPO = "Rushikeshmhaske/pneumodetect-densenet121"
MODEL_FILENAME = "pneumonia_densenet121_final.keras"


# ============================================================
# DOWNLOAD + LOAD MODEL
# ============================================================

@st.cache_resource
def load_pneumonia_model():

    try:

        model_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=MODEL_FILENAME
        )

        loaded_model = tf.keras.models.load_model(
            model_path
        )

        return loaded_model

    except Exception as e:

        st.error(
            "Unable to load the PneumoDetect AI model."
        )

        st.exception(e)

        st.stop()


model = load_pneumonia_model()


# ============================================================
# GRAD-CAM
# ============================================================

def generate_gradcam(img_array):

    base_model = model.get_layer(
        "densenet121"
    )

    global_pool = model.get_layer(
        "global_average_pooling2d_2"
    )

    dense_256 = model.get_layer(
        "dense_4"
    )

    batch_norm = model.get_layer(
        "batch_normalization_8"
    )

    dense_128 = model.get_layer(
        "dense_5"
    )

    final_dense = model.get_layer(
        "dense_6"
    )

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

        prediction = final_dense(x)

        class_score = prediction[:, 0]

    grads = tape.gradient(
        class_score,
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

    max_value = tf.reduce_max(
        heatmap
    )

    heatmap = tf.where(
        max_value > 0,
        heatmap / max_value,
        heatmap
    )

    return heatmap.numpy()


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):

    image = image.convert("RGB")

    display_image = np.array(
        image
    )

    resized = image.resize(
        (224, 224)
    )

    img_array = np.array(
        resized
    ).astype("float32") / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    return display_image, img_array


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("""
    <div class="logo-box">

        <div class="logo-icon">
            🫁
        </div>

        <div>

            <div class="logo-title">
                PneumoDetect AI
            </div>

            <div class="logo-subtitle">
                CHEST X-RAY INTELLIGENCE
            </div>

        </div>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

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
    **AI SYSTEM**

    DenseNet121  
    Transfer Learning  
    Fine-Tuned Model

    **Classes**

    • Normal  
    • Pneumonia
    """)


# ============================================================
# DASHBOARD
# ============================================================

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
            PneumoDetect AI uses a fine-tuned DenseNet121 deep learning
            model to analyze chest X-ray images and provide an
            AI-assisted pneumonia screening result with explainable
            Grad-CAM visualization.
        </p>

    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">'
        'Model Performance'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Evaluated on an untouched test dataset.'
        '</div>',
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
        '<div class="section-title">'
        'How It Works'
        '</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown("""
        ### 01 · Upload

        Upload a chest X-ray or capture an image
        using your camera.
        """)

    with c2:

        st.markdown("""
        ### 02 · Analyze

        DenseNet121 processes the X-ray and
        predicts the class.
        """)

    with c3:

        st.markdown("""
        ### 03 · Explain

        Grad-CAM highlights image regions that
        influenced the prediction.
        """)


# ============================================================
# DETECTION
# ============================================================

elif page == "🩻 Pneumonia Detection":

    st.markdown("""
    <div class="hero">

        <div class="hero-badge">
            CHEST X-RAY ANALYSIS
        </div>

        <h1>
            AI Pneumonia Screening
        </h1>

        <p>
            Upload a chest X-ray or capture one using your camera.
            The AI model will analyze the image and provide an
            explainable screening result.
        </p>

    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            '<div class="section-title">'
            'Upload X-Ray'
            '</div>',
            unsafe_allow_html=True
        )

        uploaded_file = st.file_uploader(
            "Choose a chest X-ray image",
            type=[
                "jpg",
                "jpeg",
                "png"
            ],
            help="Upload a clear chest X-ray image."
        )

    with col2:

        st.markdown(
            '<div class="section-title">'
            'Take Photo'
            '</div>',
            unsafe_allow_html=True
        )

        camera_file = st.camera_input(
            "Capture chest X-ray"
        )

    image_source = (
        uploaded_file
        if uploaded_file is not None
        else camera_file
    )

    if image_source:

        image = Image.open(
            image_source
        )

        display_image, img_array = preprocess_image(
            image
        )

        st.markdown(
            '<div class="section-title">'
            'Image Preview'
            '</div>',
            unsafe_allow_html=True
        )

        preview_col1, preview_col2, preview_col3 = st.columns(
            [1, 2, 1]
        )

        with preview_col2:

            st.image(
                display_image,
                caption="Selected Chest X-Ray",
                use_container_width=True
            )

        analyze = st.button(
            "🔍  ANALYZE X-RAY",
            type="primary",
            use_container_width=True
        )

        if analyze:

            with st.spinner(
                "Analyzing chest X-ray..."
            ):

                prediction = model.predict(
                    img_array,
                    verbose=0
                )[0][0]

                if prediction >= 0.5:

                    predicted_class = "PNEUMONIA"

                    confidence = prediction

                else:

                    predicted_class = "NORMAL"

                    confidence = 1 - prediction

                heatmap = generate_gradcam(
                    img_array
                )

            st.markdown(
                '<div class="section-title">'
                'AI Analysis Result'
                '</div>',
                unsafe_allow_html=True
            )

            if predicted_class == "PNEUMONIA":

                result_class = "result-pneumonia"

            else:

                result_class = "result-normal"

            st.markdown(
                f"""
                <div class="result-card">

                    <div class="result-title">
                        AI SCREENING RESULT
                    </div>

                    <div class="{result_class}">
                        {predicted_class}
                    </div>

                    <br>

                    <div class="result-title">
                        CONFIDENCE
                    </div>

                    <h2>
                        {confidence * 100:.2f}%
                    </h2>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.progress(
                float(confidence)
            )

            # ==================================================
            # GRAD-CAM VISUALIZATION
            # ==================================================

            original = display_image.copy()

            heatmap_resized = cv2.resize(
                heatmap,
                (
                    original.shape[1],
                    original.shape[0]
                )
            )

            heatmap_uint8 = np.uint8(
                255 * heatmap_resized
            )

            heatmap_color = cv2.applyColorMap(
                heatmap_uint8,
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

            st.markdown(
                '<div class="section-title">'
                'Explainable AI — Grad-CAM'
                '</div>',
                unsafe_allow_html=True
            )

            g1, g2, g3 = st.columns(3)

            with g1:

                st.image(
                    original,
                    caption="Original X-Ray",
                    use_container_width=True
                )

            with g2:

                st.image(
                    heatmap_resized,
                    caption="AI Attention Heatmap",
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
            <div class="disclaimer">

            ⚠️ <b>Medical Disclaimer:</b>

            This application is an academic AI research prototype.
            It is intended for educational and screening research only
            and must not be used as a substitute for professional
            medical diagnosis or clinical decision-making.

            </div>
            """, unsafe_allow_html=True)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "📊 Model Performance":

    st.markdown(
        '<div class="section-title">'
        'Model Performance'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Final evaluation on 624 untouched test images.'
        '</div>',
        unsafe_allow_html=True
    )

    metrics = {
        "Accuracy": "89.10%",
        "ROC-AUC": "94.78%",
        "Normal F1": "85.65%",
        "Pneumonia F1": "91.21%",
        "Normal Recall": "86.75%",
        "Pneumonia Recall": "90.51%"
    }

    cols = st.columns(3)

    for i, (name, value) in enumerate(
        metrics.items()
    ):

        with cols[i % 3]:

            st.markdown(
                f"""
                <div class="metric-card"
                     style="margin-bottom:18px;">

                    <div class="metric-label">
                        {name}
                    </div>

                    <div class="metric-value">
                        {value}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("""
    ### Model Architecture

    **DenseNet121 + Transfer Learning + Fine-Tuning**

    The model uses ImageNet-pretrained DenseNet121 as the
    feature extractor and a custom classification head for
    binary classification.

    **Classes**

    - NORMAL
    - PNEUMONIA
    """)


# ============================================================
# EXPLAINABLE AI
# ============================================================

elif page == "🔬 Explainable AI":

    st.markdown("""
    <div class="hero">

        <div class="hero-badge">
            EXPLAINABLE ARTIFICIAL INTELLIGENCE
        </div>

        <h1>
            Understanding the AI Decision
        </h1>

        <p>
            Grad-CAM helps visualize the image regions that
            contributed to the model's prediction.
        </p>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### What is Grad-CAM?

    Grad-CAM (Gradient-weighted Class Activation Mapping) is
    an explainability technique that creates a visual heatmap
    showing which regions of an image had greater influence
    on the model's prediction.

    **Important:** The heatmap represents model attention and
    should not be interpreted as a clinical diagnosis.
    """)

    st.info(
        "Upload an X-ray from the Detection page to generate "
        "a prediction and Grad-CAM visualization."
    )


# ============================================================
# ABOUT
# ============================================================

elif page == "ℹ️ About":

    st.markdown("""
    <div class="hero">

        <div class="hero-badge">
            ABOUT THE PROJECT
        </div>

        <h1>
            PneumoDetect AI
        </h1>

        <p>
            AI-Based Pneumonia Detection from Chest X-Ray Images
            using Deep Learning.
        </p>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ## Project Overview

    PneumoDetect AI is an academic deep learning project designed
    to demonstrate how computer vision and transfer learning can
    be applied to chest X-ray image classification.

    ### Technology Stack

    - Python
    - TensorFlow / Keras
    - DenseNet121
    - OpenCV
    - NumPy
    - Streamlit
    - Grad-CAM

    ### Model

    The final model is a fine-tuned DenseNet121 trained for binary
    classification of chest X-ray images into:

    **NORMAL** and **PNEUMONIA**

    ### Final Test Performance

    **Accuracy:** 89.10%

    **ROC-AUC:** 94.78%

    ### Purpose

    This system is intended for academic demonstration,
    research and learning purposes.

    It is **not a clinical diagnostic system**.
    """)

    st.markdown("""
    <div class="disclaimer">

    ⚠️ <b>Medical Disclaimer:</b>

    This application does not provide medical advice, diagnosis,
    treatment or clinical recommendations. Always consult a
    qualified healthcare professional for medical evaluation.

    </div>
    """, unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================

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
