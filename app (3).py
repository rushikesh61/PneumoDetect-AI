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
# Existing healthcare colors and structure preserved
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       GLOBAL
       ======================================================== */

    .main {
        background-color: #f5f9fb;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* ========================================================
       SIDEBAR
       ======================================================== */

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

    section[data-testid="stSidebar"] .stRadio label {
        font-size: 0.95rem;
    }

    /* ========================================================
       LOGO
       ======================================================== */

    .logo-box {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 10px 5px 15px 5px;
    }

    .logo-icon {
        font-size: 2.2rem;
    }

    .logo-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: white;
    }

    .logo-subtitle {
        font-size: 0.65rem;
        letter-spacing: 1.2px;
        color: #b8e5e8;
        margin-top: 3px;
    }

    /* ========================================================
       HERO
       ======================================================== */

    .hero-card {
        background: linear-gradient(
            135deg,
            #062c3d 0%,
            #073b4c 55%,
            #075e68 100%
        );

        padding: 32px;
        border-radius: 22px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 8px 25px rgba(7, 59, 76, 0.18);
    }

    .hero-title {
        font-size: 2.25rem;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .hero-subtitle {
        font-size: 1rem;
        color: #c8eef0;
        line-height: 1.6;
    }

    /* ========================================================
       SECTION TITLES
       ======================================================== */

    .section-title {
        color: #073b4c;
        font-size: 1.45rem;
        font-weight: 750;
        margin-top: 15px;
        margin-bottom: 15px;
    }

    .section-subtitle {
        color: #52717b;
        font-size: 0.95rem;
        margin-bottom: 20px;
    }

    /* ========================================================
       METRIC CARDS
       ======================================================== */

    .metric-card {
        background: white;
        border-radius: 16px;
        padding: 20px;
        border: 1px solid #dcebef;
        box-shadow: 0 4px 15px rgba(7, 59, 76, 0.07);
        min-height: 125px;
    }

    .metric-label {
        color: #607d86;
        font-size: 0.82rem;
        font-weight: 600;
        margin-bottom: 8px;
    }

    .metric-value {
        color: #073b4c;
        font-size: 1.75rem;
        font-weight: 800;
    }

    .metric-description {
        color: #6f8b93;
        font-size: 0.75rem;
        margin-top: 5px;
    }

    /* ========================================================
       INFO CARDS
       ======================================================== */

    .info-card {
        background: white;
        border: 1px solid #dcebef;
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 15px;
        box-shadow: 0 4px 15px rgba(7, 59, 76, 0.05);
    }

    .info-title {
        color: #073b4c;
        font-weight: 750;
        font-size: 1.05rem;
        margin-bottom: 8px;
    }

    .info-text {
        color: #58737c;
        font-size: 0.9rem;
        line-height: 1.6;
    }

    /* ========================================================
       RESULT CARDS
       ======================================================== */

    .result-card {
        background: white;
        border-radius: 18px;
        padding: 25px;
        border: 1px solid #dcebef;
        box-shadow: 0 5px 18px rgba(7, 59, 76, 0.08);
        text-align: center;
        margin-top: 15px;
    }

    .result-label {
        color: #6a858d;
        font-size: 0.85rem;
        margin-bottom: 7px;
    }

    .result-value {
        color: #073b4c;
        font-size: 2rem;
        font-weight: 800;
    }

    /* ========================================================
       DISCLAIMER
       ======================================================== */

    .disclaimer {
        background: #fff8e6;
        border: 1px solid #f1d58b;
        border-radius: 14px;
        padding: 18px;
        color: #725d25;
        font-size: 0.85rem;
        line-height: 1.6;
        margin-top: 25px;
    }

    /* ========================================================
       FOOTER
       ======================================================== */

    .footer {
        text-align: center;
        color: #789099;
        font-size: 0.78rem;
        padding: 25px 0 10px 0;
    }

    /* ========================================================
       BUTTONS
       ======================================================== */

    .stButton > button {
        border-radius: 10px;
        min-height: 42px;
        font-weight: 650;
    }

    /* ========================================================
       MOBILE RESPONSIVE
       ======================================================== */

    @media (max-width: 768px) {

        .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            padding-top: 1rem !important;
        }

        .hero-card {
            padding: 22px;
            border-radius: 17px;
        }

        .hero-title {
            font-size: 1.65rem;
            line-height: 1.25;
        }

        .hero-subtitle {
            font-size: 0.88rem;
            line-height: 1.5;
        }

        .section-title {
            font-size: 1.25rem;
        }

        .metric-card {
            margin-bottom: 12px;
            min-height: auto;
        }

        .metric-value {
            font-size: 1.5rem;
        }

        .result-value {
            font-size: 1.65rem;
        }

        .info-card {
            padding: 17px;
        }

        .info-text {
            font-size: 0.85rem;
        }

        [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
        }

        [data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
        }

        .stButton > button {
            width: 100% !important;
        }

        [data-testid="stFileUploader"] {
            width: 100% !important;
        }

        [data-testid="stImage"] img {
            max-width: 100% !important;
            height: auto !important;
        }

        .hero-card,
        .metric-card,
        .info-card,
        .result-card {
            max-width: 100% !important;
            box-sizing: border-box !important;
            overflow-wrap: break-word !important;
            word-wrap: break-word !important;
        }

        h1 {
            font-size: 1.8rem !important;
        }

        h2 {
            font-size: 1.4rem !important;
        }

        h3 {
            font-size: 1.15rem !important;
        }

        p {
            line-height: 1.5 !important;
        }
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HUGGING FACE MODEL
# ============================================================

MODEL_REPO = "Rushikeshmhaske/pneumodetect-densenet121"
MODEL_FILENAME = "pneumonia_densenet121_final.keras"


@st.cache_resource
def load_pneumonia_model():

    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILENAME
    )

    loaded_model = tf.keras.models.load_model(
        model_path
    )

    return loaded_model


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model = load_pneumonia_model()

except Exception as e:

    st.error(
        "Unable to load the PneumoDetect AI model."
    )

    st.code(
        str(e)
    )

    st.stop()


# ============================================================
# CHEST X-RAY VALIDATION
# ============================================================

def is_likely_chest_xray(image):

    """
    Basic application-level validation.

    This rejects obvious invalid images such as:
    - strongly colored photographs
    - extremely dark/bright images
    - very low-information images
    - unusually complex images

    This is NOT a medical-grade X-ray detector.
    """

    try:

        rgb = np.array(
            image.convert("RGB")
        )

        width, height = image.size

        # ----------------------------------------------------
        # Resolution
        # ----------------------------------------------------

        if width < 150 or height < 150:

            return (
                False,
                "Image resolution is too low."
            )

        # ----------------------------------------------------
        # Grayscale
        # ----------------------------------------------------

        gray = cv2.cvtColor(
            rgb,
            cv2.COLOR_RGB2GRAY
        )

        gray = cv2.resize(
            gray,
            (224, 224)
        )

        # ----------------------------------------------------
        # Color detection
        # ----------------------------------------------------

        channels = rgb.astype(
            np.float32
        )

        r_mean = np.mean(
            channels[:, :, 0]
        )

        g_mean = np.mean(
            channels[:, :, 1]
        )

        b_mean = np.mean(
            channels[:, :, 2]
        )

        color_difference = (
            abs(r_mean - g_mean)
            + abs(g_mean - b_mean)
            + abs(r_mean - b_mean)
        )

        if color_difference > 45:

            return (
                False,
                "The uploaded image appears to be "
                "a colored photograph, not a chest X-ray."
            )

        # ----------------------------------------------------
        # Contrast
        # ----------------------------------------------------

        std_value = np.std(
            gray
        )

        if std_value < 12:

            return (
                False,
                "The image does not contain enough "
                "visual information."
            )

        # ----------------------------------------------------
        # Edge structure
        # ----------------------------------------------------

        edges = cv2.Canny(
            gray,
            50,
            150
        )

        edge_ratio = np.mean(
            edges > 0
        )

        if edge_ratio < 0.005:

            return (
                False,
                "Insufficient image structure detected."
            )

        if edge_ratio > 0.45:

            return (
                False,
                "The image contains unusually high "
                "visual complexity."
            )

        # ----------------------------------------------------
        # Brightness
        # ----------------------------------------------------

        center = gray[
            30:194,
            30:194
        ]

        center_mean = np.mean(
            center
        )

        if center_mean < 15:

            return (
                False,
                "The image is too dark."
            )

        if center_mean > 245:

            return (
                False,
                "The image is too bright."
            )

        return (
            True,
            "Image passed basic chest X-ray validation."
        )

    except Exception:

        return (
            False,
            "Unable to validate the image."
        )


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):

    image = image.convert(
        "RGB"
    )

    image = image.resize(
        (224, 224)
    )

    image_array = np.array(
        image
    ).astype(
        np.float32
    )

    image_array = image_array / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ============================================================
# GRAD-CAM
# ============================================================

def generate_gradcam(img_array):

    try:

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

            conv_outputs, features = (
                grad_model(
                    img_array,
                    training=False
                )
            )

            x = global_pool(
                features
            )

            x = dense_256(
                x
            )

            x = batch_norm(
                x,
                training=False
            )

            x = dense_128(
                x
            )

            x = final_dense(
                x
            )

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

        if float(max_heatmap) > 0:

            heatmap = (
                heatmap / max_heatmap
            )

        return (
            heatmap.numpy(),
            float(
                prediction.numpy()[0]
            )
        )

    except Exception:

        return (
            None,
            None
        )


# ============================================================
# GRAD-CAM OVERLAY
# ============================================================

def create_gradcam_overlay(
    image,
    heatmap
):

    original = np.array(
        image.convert("RGB")
    )

    original = cv2.resize(
        original,
        (224, 224)
    )

    heatmap_uint8 = np.uint8(
        255 * heatmap
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
        0.6,
        heatmap_color,
        0.4,
        0
    )

    return overlay


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.html(
        """
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
        """
    )

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

    st.html(
        """
        <div style="
            padding: 12px;
            border-radius: 12px;
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.12);
        ">

            <div style="
                font-size: 0.75rem;
                color: #b8e5e8;
                margin-bottom: 8px;
            ">
                AI SYSTEM
            </div>

            <div style="
                font-size: 1rem;
                font-weight: 700;
                color: white;
            ">
                DenseNet121
            </div>

            <div style="
                font-size: 0.75rem;
                color: #c8eef0;
                margin-top: 4px;
            ">
                Deep Learning Classification
            </div>

            <div style="
                font-size: 0.75rem;
                color: #c8eef0;
                margin-top: 8px;
            ">
                Classes: 2
            </div>

            <div style="
                font-size: 0.75rem;
                color: #c8eef0;
            ">
                NORMAL / PNEUMONIA
            </div>

        </div>
        """
    )


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.html(
        """
        <div class="hero-card">

            <div class="hero-title">
                PneumoDetect AI
            </div>

            <div class="hero-subtitle">
                AI-powered pneumonia detection from chest
                X-ray images using Deep Learning and
                DenseNet121 transfer learning.
            </div>

        </div>
        """
    )

    st.html(
        """
        <div class="section-title">
            AI Model Overview
        </div>

        <div class="section-subtitle">
            A deep learning system designed to classify
            chest X-ray images into Normal and Pneumonia.
        </div>
        """
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.html(
            """
            <div class="metric-card">

                <div class="metric-label">
                    TEST ACCURACY
                </div>

                <div class="metric-value">
                    89.10%
                </div>

                <div class="metric-description">
                    Unseen test dataset
                </div>

            </div>
            """
        )

    with col2:

        st.html(
            """
            <div class="metric-card">

                <div class="metric-label">
                    ROC-AUC
                </div>

                <div class="metric-value">
                    94.78%
                </div>

                <div class="metric-description">
                    Classification quality
                </div>

            </div>
            """
        )

    with col3:

        st.html(
            """
            <div class="metric-card">

                <div class="metric-label">
                    PNEUMONIA RECALL
                </div>

                <div class="metric-value">
                    90.51%
                </div>

                <div class="metric-description">
                    Positive class recall
                </div>

            </div>
            """
        )

    with col4:

        st.html(
            """
            <div class="metric-card">

                <div class="metric-label">
                    PNEUMONIA F1
                </div>

                <div class="metric-value">
                    91.21%
                </div>

                <div class="metric-description">
                    Precision-recall balance
                </div>

            </div>
            """
        )

    st.markdown("")

    col1, col2 = st.columns(2)

    with col1:

        st.html(
            """
            <div class="info-card">

                <div class="info-title">
                    🧠 Deep Learning Model
                </div>

                <div class="info-text">
                    PneumoDetect AI uses DenseNet121
                    pretrained on ImageNet and fine-tuned
                    for binary chest X-ray classification.
                </div>

            </div>
            """
        )

    with col2:

        st.html(
            """
            <div class="info-card">

                <div class="info-title">
                    🔬 Explainable AI
                </div>

                <div class="info-text">
                    Grad-CAM highlights image regions that
                    contributed to the model prediction,
                    improving model interpretability.
                </div>

            </div>
            """
        )

    st.html(
        """
        <div class="info-card">

            <div class="info-title">
                🩻 Supported Predictions
            </div>

            <div class="info-text">

                <b>NORMAL</b> — No pneumonia pattern detected
                by the model.

                <br><br>

                <b>PNEUMONIA</b> — Image contains visual
                patterns associated with pneumonia according
                to the trained model.

            </div>

        </div>
        """
    )

    st.html(
        """
        <div class="disclaimer">

            <b>⚠️ Medical Disclaimer</b><br><br>

            PneumoDetect AI is an educational and research
            project. It is not a medical device and should
            not be used as a substitute for professional
            medical diagnosis.

        </div>
        """
    )


# ============================================================
# PNEUMONIA DETECTION
# ============================================================

elif page == "🩻 Pneumonia Detection":

    st.html(
        """
        <div class="hero-card">

            <div class="hero-title">
                🩻 Pneumonia Detection
            </div>

            <div class="hero-subtitle">
                Upload a chest X-ray or use your device camera
                to perform AI-based pneumonia classification.
            </div>

        </div>
        """
    )

    st.html(
        """
        <div class="section-title">
            Select Image Source
        </div>

        <div class="section-subtitle">
            Choose how you want to provide the chest X-ray.
        </div>
        """
    )

    # ========================================================
    # IMPORTANT:
    # Camera does NOT open automatically.
    # ========================================================

    input_method = st.radio(
        "Image Source",
        [
            "📁 Upload Image",
            "📷 Use Camera"
        ],
        horizontal=True
    )

    image = None

    # ========================================================
    # UPLOAD
    # ========================================================

    if input_method == "📁 Upload Image":

        uploaded_file = st.file_uploader(
            "Upload Chest X-Ray Image",
            type=[
                "jpg",
                "jpeg",
                "png"
            ],
            help="Upload a clear chest X-ray image."
        )

        if uploaded_file is not None:

            image = Image.open(
                uploaded_file
            ).convert("RGB")

    # ========================================================
    # CAMERA
    # ========================================================

    elif input_method == "📷 Use Camera":

        st.info(
            "Camera will be available after selecting "
            "'Use Camera'."
        )

        camera_image = st.camera_input(
            "Take a picture of the chest X-ray"
        )

        if camera_image is not None:

            image = Image.open(
                camera_image
            ).convert("RGB")

    # ========================================================
    # IMAGE PREVIEW + ANALYSIS
    # ========================================================

    if image is not None:

        st.markdown("")

        st.html(
            """
            <div class="section-title">
                Image Preview
            </div>
            """
        )

        st.image(
            image,
            caption="Selected Chest X-Ray",
            use_container_width=True
        )

        st.markdown("")

        analyze = st.button(
            "🔍 Analyze Chest X-Ray",
            use_container_width=True
        )

        if analyze:

            # =================================================
            # STEP 1 — VALIDATE IMAGE
            # =================================================

            is_valid, validation_message = (
                is_likely_chest_xray(
                    image
                )
            )

            # =================================================
            # STEP 2 — INVALID IMAGE
            # =================================================

            if not is_valid:

                st.error(
                    "❌ Invalid Image"
                )

                st.warning(
                    "Please upload a clear chest X-ray image. "
                    "The uploaded image does not appear to be "
                    "suitable for pneumonia detection."
                )

                st.info(
                    f"Validation: {validation_message}"
                )

                st.stop()

            # =================================================
            # STEP 3 — PREPROCESS
            # =================================================

            img_array = preprocess_image(
                image
            )

            # =================================================
            # STEP 4 — MODEL PREDICTION
            # =================================================

            with st.spinner(
                "Analyzing chest X-ray..."
            ):

                prediction = model.predict(
                    img_array,
                    verbose=0
                )[0][0]

                if prediction >= 0.5:

                    predicted_class = (
                        "PNEUMONIA"
                    )

                    confidence = (
                        prediction
                    )

                else:

                    predicted_class = (
                        "NORMAL"
                    )

                    confidence = (
                        1 - prediction
                    )

            # =================================================
            # RESULT
            # =================================================

            st.markdown("")

            st.html(
                """
                <div class="section-title">
                    AI Prediction
                </div>
                """
            )

            result_col1, result_col2 = (
                st.columns(2)
            )

            with result_col1:

                st.html(
                    f"""
                    <div class="result-card">

                        <div class="result-label">
                            PREDICTION
                        </div>

                        <div class="result-value">
                            {predicted_class}
                        </div>

                    </div>
                    """
                )

            with result_col2:

                st.html(
                    f"""
                    <div class="result-card">

                        <div class="result-label">
                            CONFIDENCE
                        </div>

                        <div class="result-value">
                            {confidence * 100:.2f}%
                        </div>

                    </div>
                    """
                )

            # =================================================
            # STREAMLIT STATUS
            # =================================================

            if predicted_class == "PNEUMONIA":

                st.warning(
                    "⚠️ Model prediction: PNEUMONIA"
                )

                st.write(
                    "The model detected visual patterns "
                    "associated with pneumonia."
                )

            else:

                st.success(
                    "✅ Model prediction: NORMAL"
                )

                st.write(
                    "The model did not detect visual patterns "
                    "associated with pneumonia."
                )

            # =================================================
            # GRAD-CAM
            # =================================================

            with st.spinner(
                "Generating Grad-CAM explanation..."
            ):

                heatmap, gradcam_prediction = (
                    generate_gradcam(
                        img_array
                    )
                )

            if heatmap is not None:

                overlay = (
                    create_gradcam_overlay(
                        image,
                        heatmap
                    )
                )

                st.markdown("")

                st.html(
                    """
                    <div class="section-title">
                        🔬 Explainable AI — Grad-CAM
                    </div>

                    <div class="section-subtitle">
                        Highlighted regions show areas that
                        contributed to the model prediction.
                    </div>
                    """
                )

                cam_col1, cam_col2 = (
                    st.columns(2)
                )

                with cam_col1:

                    st.image(
                        image,
                        caption="Original Chest X-Ray",
                        use_container_width=True
                    )

                with cam_col2:

                    st.image(
                        overlay,
                        caption="Grad-CAM Heatmap Overlay",
                        use_container_width=True
                    )

            else:

                st.info(
                    "Grad-CAM explanation could not be "
                    "generated for this image."
                )

            # =================================================
            # DISCLAIMER
            # =================================================

            st.html(
                """
                <div class="disclaimer">

                    <b>⚠️ Important</b><br><br>

                    This prediction is generated by an AI
                    research model and is intended only for
                    educational and demonstration purposes.

                    It should not be considered a medical
                    diagnosis. Always consult a qualified
                    healthcare professional.

                </div>
                """
            )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "📊 Model Performance":

    st.html(
        """
        <div class="hero-card">

            <div class="hero-title">
                📊 Model Performance
            </div>

            <div class="hero-subtitle">
                Performance of the fine-tuned DenseNet121
                model on the untouched test dataset.
            </div>

        </div>
        """
    )

    st.html(
        """
        <div class="section-title">
            Final Test Performance
        </div>
        """
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.html(
            """
            <div class="metric-card">

                <div class="metric-label">
                    ACCURACY
                </div>

                <div class="metric-value">
                    89.10%
                </div>

            </div>
            """
        )

    with col2:

        st.html(
            """
            <div class="metric-card">

                <div class="metric-label">
                    PRECISION
                </div>

                <div class="metric-value">
                    91.93%
                </div>

            </div>
            """
        )

    with col3:

        st.html(
            """
            <div class="metric-card">

                <div class="metric-label">
                    RECALL
                </div>

                <div class="metric-value">
                    90.51%
                </div>

            </div>
            """
        )

    with col4:

        st.html(
            """
            <div class="metric-card">

                <div class="metric-label">
                    ROC-AUC
                </div>

                <div class="metric-value">
                    94.78%
                </div>

            </div>
            """
        )

    st.markdown("")

    st.html(
        """
        <div class="section-title">
            Classification Report
        </div>
        """
    )

    performance_data = {
        "Class": [
            "NORMAL",
            "PNEUMONIA",
            "Macro Average",
            "Weighted Average"
        ],

        "Precision": [
            "84.58%",
            "91.93%",
            "88.26%",
            "89.17%"
        ],

        "Recall": [
            "86.75%",
            "90.51%",
            "88.63%",
            "89.10%"
        ],

        "F1 Score": [
            "85.65%",
            "91.21%",
            "88.43%",
            "89.13%"
        ]
    }

    st.table(
        performance_data
    )

    st.markdown("")

    st.html(
        """
        <div class="section-title">
            Confusion Matrix
        </div>

        <div class="info-card">

            <div class="info-title">
                Test Set Results
            </div>

            <div class="info-text">

                <b>True Normal:</b> 203<br>
                <b>False Pneumonia:</b> 31<br>
                <b>False Normal:</b> 37<br>
                <b>True Pneumonia:</b> 353

            </div>

        </div>
        """
    )

    st.html(
        """
        <div class="section-title">
            Model Comparison
        </div>
        """
    )

    comparison_data = {
        "Model": [
            "Custom CNN",
            "DenseNet121 + Fine-Tuning"
        ],

        "Test Accuracy": [
            "69.87%",
            "89.10%"
        ],

        "ROC-AUC": [
            "86.33%",
            "94.78%"
        ]
    }

    st.table(
        comparison_data
    )

    st.success(
        "DenseNet121 improved test accuracy by "
        "19.23 percentage points compared with "
        "the custom CNN baseline."
    )


# ============================================================
# EXPLAINABLE AI
# ============================================================

elif page == "🔬 Explainable AI":

    st.html(
        """
        <div class="hero-card">

            <div class="hero-title">
                🔬 Explainable AI
            </div>

            <div class="hero-subtitle">
                Understanding which image regions influence
                the PneumoDetect AI prediction.
            </div>

        </div>
        """
    )

    col1, col2 = st.columns(2)

    with col1:

        st.html(
            """
            <div class="info-card">

                <div class="info-title">
                    What is Grad-CAM?
                </div>

                <div class="info-text">

                    Grad-CAM stands for Gradient-weighted
                    Class Activation Mapping.

                    <br><br>

                    It uses gradients from the deep learning
                    model to identify image regions that are
                    important for the final prediction.

                </div>

            </div>
            """
        )

    with col2:

        st.html(
            """
            <div class="info-card">

                <div class="info-title">
                    Why is it useful?
                </div>

                <div class="info-text">

                    Deep learning models can behave like
                    black boxes.

                    <br><br>

                    Grad-CAM provides visual information about
                    where the model is focusing, improving
                    interpretability and transparency.

                </div>

            </div>
            """
        )

    st.html(
        """
        <div class="info-card">

            <div class="info-title">
                DenseNet121 Feature Layer
            </div>

            <div class="info-text">

                Grad-CAM is generated from the final convolutional
                layer:

                <br><br>

                <b>conv5_block16_2_conv</b>

                <br><br>

                The generated heatmap is resized and overlaid
                on the original chest X-ray.

            </div>

        </div>
        """
    )

    st.html(
        """
        <div class="info-card">

            <div class="info-title">
                Interpretation
            </div>

            <div class="info-text">

                🔴 Red / warm regions indicate areas with
                stronger contribution to the prediction.

                <br><br>

                🔵 Blue / cool regions indicate areas with
                relatively lower contribution.

                <br><br>

                Grad-CAM should be interpreted as an AI
                explanation, not as a clinical diagnosis.

            </div>

        </div>
        """
    )


# ============================================================
# ABOUT
# ============================================================

elif page == "ℹ️ About":

    st.html(
        """
        <div class="hero-card">

            <div class="hero-title">
                ℹ️ About PneumoDetect AI
            </div>

            <div class="hero-subtitle">
                Deep Learning based chest X-ray
                pneumonia classification system.
            </div>

        </div>
        """
    )

    st.html(
        """
        <div class="info-card">

            <div class="info-title">
                🎯 Project Objective
            </div>

            <div class="info-text">

                The objective of PneumoDetect AI is to
                demonstrate how Deep Learning and transfer
                learning can be used for automated
                classification of chest X-ray images.

            </div>

        </div>
        """
    )

    col1, col2 = st.columns(2)

    with col1:

        st.html(
            """
            <div class="info-card">

                <div class="info-title">
                    🧠 Model
                </div>

                <div class="info-text">

                    <b>Architecture:</b> DenseNet121<br><br>

                    <b>Learning:</b> Transfer Learning +
                    Fine-Tuning<br><br>

                    <b>Input:</b> 224 × 224 RGB image<br><br>

                    <b>Output:</b> Binary classification

                </div>

            </div>
            """
        )

    with col2:

        st.html(
            """
            <div class="info-card">

                <div class="info-title">
                    🩻 Classes
                </div>

                <div class="info-text">

                    <b>Class 0:</b> NORMAL<br><br>

                    <b>Class 1:</b> PNEUMONIA<br><br>

                    The model predicts the probability
                    of pneumonia from the uploaded image.

                </div>

            </div>
            """
        )

    st.html(
        """
        <div class="info-card">

            <div class="info-title">
                🛠️ Technology Stack
            </div>

            <div class="info-text">

                <b>Python</b> — Programming Language<br><br>

                <b>TensorFlow / Keras</b> — Deep Learning<br><br>

                <b>DenseNet121</b> — Transfer Learning<br><br>

                <b>OpenCV</b> — Image Processing<br><br>

                <b>Streamlit</b> — Web Application<br><br>

                <b>Hugging Face</b> — Model Hosting<br><br>

                <b>Grad-CAM</b> — Explainable AI

            </div>

        </div>
        """
    )

    st.html(
        """
        <div class="info-card">

            <div class="info-title">
                📈 Final Model Results
            </div>

            <div class="info-text">

                Test Accuracy: <b>89.10%</b><br>
                ROC-AUC: <b>94.78%</b><br>
                Pneumonia Precision: <b>91.93%</b><br>
                Pneumonia Recall: <b>90.51%</b><br>
                Pneumonia F1 Score: <b>91.21%</b>

            </div>

        </div>
        """
    )

    st.html(
        """
        <div class="disclaimer">

            <b>⚠️ Medical Disclaimer</b><br><br>

            PneumoDetect AI is a college-level Deep Learning
            research and demonstration project.

            <br><br>

            It is not intended for clinical diagnosis,
            treatment decisions, or emergency medical use.

            <br><br>

            Always consult a qualified healthcare professional
            for medical interpretation of chest X-rays.

        </div>
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.html(
    """
    <div class="footer">

        PneumoDetect AI • DenseNet121 • Deep Learning
        • Explainable AI

    </div>
    """
)
