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

st.markdown(
    """
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
        radial-gradient(
            circle at 10% 10%,
            rgba(13,148,136,0.08),
            transparent 25%
        ),
        radial-gradient(
            circle at 90% 20%,
            rgba(14,116,144,0.07),
            transparent 25%
        ),
        #f7fafc;
    }

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

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

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
        background: linear-gradient(
            135deg,
            #14b8a6,
            #0f766e
        );
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

    .disclaimer {
        padding: 18px 20px;
        border-radius: 15px;
        background: #fff7ed;
        border: 1px solid #fed7aa;
        color: #7c2d12;
        font-size: 13px;
        line-height: 1.6;
    }

    .footer {
        margin-top: 50px;
        padding-top: 20px;
        border-top: 1px solid #e2e8f0;
        text-align: center;
        color: #94a3b8;
        font-size: 12px;
    }

    /* ========================================================
       MOBILE RESPONSIVE
       Colors and desktop structure are unchanged
       ======================================================== */

    @media (max-width: 768px) {

        .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            padding-top: 1rem !important;
        }

        .hero {
            padding: 24px;
            border-radius: 20px;
        }

        .hero h1 {
            font-size: 28px;
            line-height: 1.2;
        }

        .hero p {
            font-size: 14px;
            line-height: 1.6;
        }

        .section-title {
            font-size: 21px;
        }

        .metric-card {
            margin-bottom: 14px;
        }

        .metric-value {
            font-size: 25px;
        }

        .result-card {
            margin-bottom: 15px;
            padding: 22px;
        }

        .result-pneumonia,
        .result-normal {
            font-size: 28px;
        }

        /* Stack columns on mobile */
        [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
        }

        [data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
            min-width: 100% !important;
        }

        /* Images stay inside screen */
        [data-testid="stImage"] img {
            max-width: 100% !important;
            height: auto !important;
        }

        /* Buttons */
        .stButton > button {
            width: 100% !important;
            min-height: 44px !important;
        }

        /* Uploader */
        [data-testid="stFileUploader"] {
            width: 100% !important;
        }

        /* Prevent text overflow */
        .hero,
        .metric-card,
        .result-card,
        .disclaimer {
            overflow-wrap: break-word !important;
            word-wrap: break-word !important;
        }

        /* Fix invisible text inside white cards on mobile */
        .metric-card h3,
        .metric-card p {
            color: #073b4c !important;
            opacity: 1 !important;
            visibility: visible !important;
        }

        .metric-card p {
            color: #64748b !important;
            line-height: 1.6 !important;
        }

        .metric-card h3 {
            color: #073b4c !important;
            line-height: 1.35 !important;
        }
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_REPO = "Rushikeshmhaske/pneumodetect-densenet121"

MODEL_FILENAME = "pneumonia_densenet121_final.keras"


# ============================================================
# LOAD MODEL FROM HUGGING FACE
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
# CHEST X-RAY IMAGE VALIDATION
# ============================================================

def is_likely_chest_xray(image):

    """
    Basic validation to reject obvious non-X-ray images.

    This is NOT a medical-grade X-ray detector.
    It is an app-level safety filter.
    """

    try:

        # Convert to RGB
        rgb = np.array(
            image.convert("RGB")
        )

        # Image dimensions
        width, height = image.size

        if width < 150 or height < 150:

            return (
                False,
                "Image resolution is too low."
            )

        # Convert to grayscale
        gray = cv2.cvtColor(
            rgb,
            cv2.COLOR_RGB2GRAY
        )

        # Resize
        gray = cv2.resize(
            gray,
            (224, 224)
        )

        # ----------------------------------------------------
        # COLOR CHECK
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

        channel_difference = (
            abs(r_mean - g_mean)
            + abs(g_mean - b_mean)
            + abs(r_mean - b_mean)
        )

        if channel_difference > 45:

            return (
                False,
                "This does not appear to be a chest X-ray."
            )

        # ----------------------------------------------------
        # GRAYSCALE VARIATION
        # ----------------------------------------------------

        std_value = np.std(
            gray
        )

        if std_value < 12:

            return (
                False,
                "Image does not contain enough "
                "X-ray-like visual information."
            )

        # ----------------------------------------------------
        # EDGE STRUCTURE
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
                "No sufficient medical image structure detected."
            )

        if edge_ratio > 0.45:

            return (
                False,
                "Image contains unusually high visual complexity."
            )

        # ----------------------------------------------------
        # CENTRAL BRIGHTNESS
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
                "Image intensity is too dark."
            )

        if center_mean > 245:

            return (
                False,
                "Image intensity is too bright."
            )

        return (
            True,
            "Image passed the basic chest X-ray validation."
        )

    except Exception:

        return (
            False,
            "Unable to validate the uploaded image."
        )


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):

    image = image.convert(
        "RGB"
    )

    display_image = np.array(
        image
    )

    resized = image.resize(
        (224, 224)
    )

    img_array = np.array(
        resized
    ).astype(
        "float32"
    ) / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    return (
        display_image,
        img_array
    )


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

            prediction = final_dense(
                x
            )

            class_score = prediction[:, 0]

        grads = tape.gradient(
            class_score,
            conv_outputs
        )

        if grads is None:

            return None

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

        if float(max_value) > 0:

            heatmap = (
                heatmap / max_value
            )

        return heatmap.numpy()

    except Exception:

        return None


# ============================================================
# SAFE GRAD-CAM OVERLAY
# ============================================================

def create_gradcam_overlay(
    original_image,
    heatmap
):

    try:

        # ----------------------------------------------------
        # Convert original image to uint8 RGB
        # ----------------------------------------------------

        original = np.asarray(
            original_image
        )

        if original.ndim == 2:

            original = cv2.cvtColor(
                original,
                cv2.COLOR_GRAY2RGB
            )

        elif original.shape[-1] == 4:

            original = cv2.cvtColor(
                original,
                cv2.COLOR_RGBA2RGB
            )

        original = np.ascontiguousarray(
            original
        )

        if original.dtype != np.uint8:

            if original.max() <= 1.0:

                original = (
                    original * 255
                ).astype(
                    np.uint8
                )

            else:

                original = np.clip(
                    original,
                    0,
                    255
                ).astype(
                    np.uint8
                )

        # ----------------------------------------------------
        # Resize original to fixed display size
        # ----------------------------------------------------

        original = cv2.resize(
            original,
            (224, 224),
            interpolation=cv2.INTER_AREA
        )

        # ----------------------------------------------------
        # Convert heatmap to float32
        # ----------------------------------------------------

        heatmap = np.asarray(
            heatmap,
            dtype=np.float32
        )

        # Remove unnecessary dimensions
        heatmap = np.squeeze(
            heatmap
        )

        # Make sure heatmap is 2D
        if heatmap.ndim != 2:

            return None

        # ----------------------------------------------------
        # Resize heatmap EXACTLY to original dimensions
        # ----------------------------------------------------

        heatmap = cv2.resize(
            heatmap,
            (
                original.shape[1],
                original.shape[0]
            ),
            interpolation=cv2.INTER_LINEAR
        )

        # Normalize safely
        heatmap = np.nan_to_num(
            heatmap,
            nan=0.0,
            posinf=1.0,
            neginf=0.0
        )

        heatmap = np.clip(
            heatmap,
            0.0,
            1.0
        )

        # ----------------------------------------------------
        # Convert heatmap to uint8
        # ----------------------------------------------------

        heatmap_uint8 = np.uint8(
            heatmap * 255
        )

        # ----------------------------------------------------
        # Apply color map
        # ----------------------------------------------------

        heatmap_color = cv2.applyColorMap(
            heatmap_uint8,
            cv2.COLORMAP_JET
        )

        heatmap_color = cv2.cvtColor(
            heatmap_color,
            cv2.COLOR_BGR2RGB
        )

        # Ensure same type
        heatmap_color = np.ascontiguousarray(
            heatmap_color
        ).astype(
            np.uint8
        )

        # ----------------------------------------------------
        # FINAL SAFETY CHECK
        # ----------------------------------------------------

        if (
            original.shape !=
            heatmap_color.shape
        ):

            heatmap_color = cv2.resize(
                heatmap_color,
                (
                    original.shape[1],
                    original.shape[0]
                )
            )

        # ----------------------------------------------------
        # Blend
        # ----------------------------------------------------

        overlay = cv2.addWeighted(
            original,
            0.60,
            heatmap_color,
            0.40,
            0
        )

        return overlay

    except Exception:

        return None


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
        <div>

            <b>AI SYSTEM</b>

            <br><br>

            <b>DenseNet121</b><br>

            Deep Learning Classification

            <br><br>

            Classes: 2

            <br>

            NORMAL / PNEUMONIA

        </div>
        """
    )


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.html(
        """
        <div class="hero">

            <div class="hero-badge">
                ● AI-POWERED HEALTHCARE SCREENING
            </div>

            <h1>
                Intelligent Pneumonia
                <br>
                Detection from Chest X-Rays
            </h1>

            <p>
                PneumoDetect AI uses a fine-tuned DenseNet121
                deep learning model to analyze chest X-ray
                images and provide an AI-assisted pneumonia
                screening result with explainable Grad-CAM
                visualization.
            </p>

        </div>
        """
    )

    st.html(
        """
        <div class="section-title">
            Model Performance
        </div>

        <div class="section-subtitle">
            Evaluated on an untouched test dataset.
        </div>
        """
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.html(
            """
            <div class="metric-card">

                <div class="metric-label">
                    Test Accuracy
                </div>

                <div class="metric-value">
                    89.10%
                </div>

            </div>
            """
        )

    with c2:

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

    with c3:

        st.html(
            """
            <div class="metric-card">

                <div class="metric-label">
                    Pneumonia Recall
                </div>

                <div class="metric-value">
                    90.51%
                </div>

            </div>
            """
        )

    with c4:

        st.html(
            """
            <div class="metric-card">

                <div class="metric-label">
                    Pneumonia F1
                </div>

                <div class="metric-value">
                    91.21%
                </div>

            </div>
            """
        )

    st.html(
        """
        <div class="section-title">
            How It Works
        </div>
        """
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.html(
            """
            <div class="metric-card">

                <div class="metric-label">
                    01 · UPLOAD
                </div>

                <h3>
                    Upload X-Ray
                </h3>

                <p>
                    Upload a chest X-ray or capture
                    an image using your camera.
                </p>

            </div>
            """
        )

    with c2:

        st.html(
            """
            <div class="metric-card">

                <div class="metric-label">
                    02 · ANALYZE
                </div>

                <h3>
                    Analyze
                </h3>

                <p>
                    DenseNet121 processes the X-ray
                    and predicts the class.
                </p>

            </div>
            """
        )

    with c3:

        st.html(
            """
            <div class="metric-card">

                <div class="metric-label">
                    03 · EXPLAIN
                </div>

                <h3>
                    Explain
                </h3>

                <p>
                    Grad-CAM highlights image regions
                    that influenced the prediction.
                </p>

            </div>
            """
        )


# ============================================================
# PNEUMONIA DETECTION
# ============================================================

elif page == "🩻 Pneumonia Detection":

    st.html(
        """
        <div class="hero">

            <div class="hero-badge">
                CHEST X-RAY ANALYSIS
            </div>

            <h1>
                AI Pneumonia Screening
            </h1>

            <p>
                Select an image source below. The camera
                will only be activated when you select
                the Use Camera option.
            </p>

        </div>
        """
    )

    st.html(
        """
        <div class="section-title">
            Select Image Source
        </div>

        <div class="section-subtitle">
            Choose whether you want to upload an X-ray
            or capture one using your device camera.
        </div>
        """
    )

    # ========================================================
    # IMAGE SOURCE SELECTION
    # ========================================================

    input_method = st.radio(
        "Image Source",
        [
            "📁 Upload Image",
            "📷 Use Camera"
        ],
        horizontal=True
    )

    image_source = None

    # ========================================================
    # UPLOAD IMAGE
    # ========================================================

    if input_method == "📁 Upload Image":

        uploaded_file = st.file_uploader(
            "Choose a chest X-ray image",
            type=[
                "jpg",
                "jpeg",
                "png"
            ],
            help="Upload a clear chest X-ray image."
        )

        if uploaded_file is not None:

            image_source = uploaded_file

    # ========================================================
    # CAMERA
    # ========================================================

    elif input_method == "📷 Use Camera":

        camera_file = st.camera_input(
            "Capture chest X-ray"
        )

        if camera_file is not None:

            image_source = camera_file

    # ========================================================
    # IMAGE SELECTED
    # ========================================================

    if image_source is not None:

        image = Image.open(
            image_source
        ).convert(
            "RGB"
        )

        display_image, img_array = (
            preprocess_image(
                image
            )
        )

        st.html(
            """
            <div class="section-title">
                Image Preview
            </div>
            """
        )

        st.image(
            display_image,
            caption="Selected Chest X-Ray",
            use_container_width=True
        )

        st.markdown("")

        analyze = st.button(
            "🔍 ANALYZE X-RAY",
            type="primary",
            use_container_width=True
        )

        # ====================================================
        # ANALYZE
        # ====================================================

        if analyze:

            # ------------------------------------------------
            # VALIDATE IMAGE BEFORE MODEL
            # ------------------------------------------------

            is_valid, validation_message = (
                is_likely_chest_xray(
                    image
                )
            )

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

            # ------------------------------------------------
            # MODEL PREDICTION
            # ------------------------------------------------

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

            # ------------------------------------------------
            # RESULT
            # ------------------------------------------------

            st.html(
                """
                <div class="section-title">
                    AI Analysis Result
                </div>
                """
            )

            if predicted_class == "PNEUMONIA":

                result_class = (
                    "result-pneumonia"
                )

            else:

                result_class = (
                    "result-normal"
                )

            r1, r2 = st.columns(2)

            with r1:

                st.html(
                    f"""
                    <div class="result-card">

                        <div class="result-title">
                            PREDICTION
                        </div>

                        <div class="{result_class}">
                            {predicted_class}
                        </div>

                    </div>
                    """
                )

            with r2:

                st.html(
                    f"""
                    <div class="result-card">

                        <div class="result-title">
                            CONFIDENCE
                        </div>

                        <div class="{result_class}">
                            {confidence * 100:.2f}%
                        </div>

                    </div>
                    """
                )

            st.markdown("")

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

            st.html(
                """
                <div class="section-title">
                    Explainable AI — Grad-CAM
                </div>

                <div class="section-subtitle">
                    The highlighted regions show areas that
                    contributed to the model prediction.
                </div>
                """
            )

            with st.spinner(
                "Generating Grad-CAM..."
            ):

                heatmap = generate_gradcam(
                    img_array
                )

            if heatmap is not None:

                overlay = (
                    create_gradcam_overlay(
                        display_image,
                        heatmap
                    )
                )

                if overlay is not None:

                    # Safe heatmap display
                    heatmap_display = cv2.resize(
                        heatmap,
                        (
                            224,
                            224
                        ),
                        interpolation=cv2.INTER_LINEAR
                    )

                    heatmap_display = np.clip(
                        heatmap_display,
                        0,
                        1
                    )

                    g1, g2, g3 = st.columns(3)

                    with g1:

                        st.image(
                            display_image,
                            caption="Original X-Ray",
                            use_container_width=True
                        )

                    with g2:

                        st.image(
                            heatmap_display,
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

                else:

                    st.warning(
                        "Grad-CAM overlay could not be generated."
                    )

            else:

                st.warning(
                    "Grad-CAM could not be generated "
                    "for this image."
                )

            # =================================================
            # DISCLAIMER
            # =================================================

            st.markdown("")

            st.html(
                """
                <div class="disclaimer">

                    ⚠️ <b>Medical Disclaimer:</b>

                    This application is an academic AI
                    research prototype. It is intended for
                    educational and research purposes only
                    and must not be used as a substitute for
                    professional medical diagnosis or
                    clinical decision-making.

                </div>
                """
            )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "📊 Model Performance":

    st.html(
        """
        <div class="section-title">
            Model Performance
        </div>

        <div class="section-subtitle">
            Final evaluation on 624 untouched test images.
        </div>
        """
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

            st.html(
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
                """
            )

    st.markdown("")

    st.markdown(
        """
        ### Classification Performance

        | Class | Precision | Recall | F1 Score |
        |---|---:|---:|---:|
        | NORMAL | 84.58% | 86.75% | 85.65% |
        | PNEUMONIA | 91.93% | 90.51% | 91.21% |
        | Macro Average | 88.26% | 88.63% | 88.43% |
        | Weighted Average | 89.17% | 89.10% | 89.13% |

        ### Confusion Matrix

        - True Normal: **203**
        - False Pneumonia: **31**
        - False Normal: **37**
        - True Pneumonia: **353**

        ### Model Architecture

        **DenseNet121 + Transfer Learning + Fine-Tuning**

        The model uses ImageNet-pretrained DenseNet121 as
        the feature extractor and a custom classification
        head for binary chest X-ray classification.

        **Classes**

        - NORMAL
        - PNEUMONIA
        """
    )


# ============================================================
# EXPLAINABLE AI
# ============================================================

elif page == "🔬 Explainable AI":

    st.html(
        """
        <div class="hero">

            <div class="hero-badge">
                EXPLAINABLE ARTIFICIAL INTELLIGENCE
            </div>

            <h1>
                Understanding the AI Decision
            </h1>

            <p>
                Grad-CAM helps visualize the image regions
                that contributed to the model's prediction.
            </p>

        </div>
        """
    )

    st.markdown(
        """
        ### What is Grad-CAM?

        Grad-CAM stands for **Gradient-weighted Class
        Activation Mapping**.

        It creates a visual heatmap showing image regions
        that had greater influence on the model prediction.

        ### DenseNet121 Grad-CAM Layer

        The final convolutional layer used for Grad-CAM is:

        **`conv5_block16_2_conv`**

        ### Why is it useful?

        Deep learning models can behave like black boxes.
        Grad-CAM provides a visual explanation of where the
        model is focusing.

        **Important:** Grad-CAM represents model attention.
        It is not a clinical diagnosis.
        """
    )

    st.info(
        "Upload an X-ray from the Pneumonia Detection page "
        "to generate a live Grad-CAM visualization."
    )


# ============================================================
# ABOUT
# ============================================================

elif page == "ℹ️ About":

    st.html(
        """
        <div class="hero">

            <div class="hero-badge">
                ABOUT THE PROJECT
            </div>

            <h1>
                PneumoDetect AI
            </h1>

            <p>
                AI-Based Pneumonia Detection from Chest
                X-Ray Images using Deep Learning.
            </p>

        </div>
        """
    )

    st.markdown(
        """
        ## Project Overview

        PneumoDetect AI is an academic Deep Learning project
        designed to demonstrate how computer vision and
        transfer learning can be applied to chest X-ray
        classification.

        ### Technology Stack

        - Python
        - TensorFlow / Keras
        - DenseNet121
        - OpenCV
        - NumPy
        - Streamlit
        - Hugging Face
        - Grad-CAM

        ### Model

        The final model is a fine-tuned **DenseNet121** trained
        for binary classification.

        **Class 0:** NORMAL

        **Class 1:** PNEUMONIA

        ### Final Test Performance

        **Accuracy:** 89.10%

        **ROC-AUC:** 94.78%

        **Pneumonia Precision:** 91.93%

        **Pneumonia Recall:** 90.51%

        **Pneumonia F1 Score:** 91.21%
        """
    )

    st.html(
        """
        <div class="disclaimer">

            ⚠️ <b>Medical Disclaimer:</b>

            PneumoDetect AI is an academic research and
            demonstration project.

            It is not a clinical diagnostic system and should
            not be used for medical diagnosis, treatment
            decisions or emergency medical use.

            Always consult a qualified healthcare professional
            for medical evaluation.

        </div>
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.html(
    """
    <div class="footer">

        <b>PneumoDetect AI</b>
        · Deep Learning Healthcare Research Project

        <br>

        Built with TensorFlow, DenseNet121,
        Grad-CAM & Streamlit

        <br><br>

        © 2026 PneumoDetect AI · Academic Project

    </div>
    """
)
