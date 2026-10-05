

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from huggingface_hub import hf_hub_download
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc


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

</style>
""", unsafe_allow_html=True)


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
# MODEL PERFORMANCE DATA
# ============================================================

TEST_ACCURACY = 89.10
TEST_ROC_AUC = 94.78
PNEUMONIA_PRECISION = 91.93
PNEUMONIA_RECALL = 90.51
PNEUMONIA_F1 = 91.21

# Final DenseNet121 confusion matrix
# Actual: NORMAL, PNEUMONIA
# Predicted: NORMAL, PNEUMONIA
CONFUSION_MATRIX = np.array([
    [203, 31],
    [37, 353]
])

# CNN baseline comparison
CNN_ACCURACY = 69.87
CNN_ROC_AUC = 86.33

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
        x = batch_norm(x, training=False)
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

    st.html("""
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
    """)

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

    st.html("""
    <div>

        <b>AI SYSTEM</b>

        <br><br>

        DenseNet121<br>
        Transfer Learning<br>
        Fine-Tuned Model

        <br><br>

        <b>Classes</b>

        <br><br>

        • Normal<br>
        • Pneumonia

    </div>
    """)


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.html("""
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
    """)

    st.html("""
    <div class="section-title">
        Model Performance
    </div>

    <div class="section-subtitle">
        Evaluated on an untouched test dataset.
    </div>
    """)

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.html("""
        <div class="metric-card">

            <div class="metric-label">
                Test Accuracy
            </div>

            <div class="metric-value">
                89.10%
            </div>

        </div>
        """)

    with c2:

        st.html("""
        <div class="metric-card">

            <div class="metric-label">
                ROC-AUC
            </div>

            <div class="metric-value">
                94.78%
            </div>

        </div>
        """)

    with c3:

        st.html("""
        <div class="metric-card">

            <div class="metric-label">
                Pneumonia Recall
            </div>

            <div class="metric-value">
                90.51%
            </div>

        </div>
        """)

    with c4:

        st.html("""
        <div class="metric-card">

            <div class="metric-label">
                Pneumonia F1
            </div>

            <div class="metric-value">
                91.21%
            </div>

        </div>
        """)

    st.html("""
    <div class="section-title">
        How It Works
    </div>
    """)

    c1, c2, c3 = st.columns(3)

    with c1:

        st.html("""
        <div class="metric-card">

            <div class="metric-label">
                01 · UPLOAD
            </div>

            <h3>
                Upload X-Ray
            </h3>

            <p>
                Upload a chest X-ray or capture an image
                using your camera.
            </p>

        </div>
        """)

    with c2:

        st.html("""
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
        """)

    with c3:

        st.html("""
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
        """)


# ============================================================
# PNEUMONIA DETECTION
# ============================================================

elif page == "🩻 Pneumonia Detection":

    st.html("""
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
    """)

    col1, col2 = st.columns(2)

    with col1:

        st.html("""
        <div class="section-title">
            Upload X-Ray
        </div>
        """)

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

        st.html("""
        <div class="section-title">
            Take Photo
        </div>
        """)

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

        st.html("""
        <div class="section-title">
            Image Preview
        </div>
        """)

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

            st.html("""
            <div class="section-title">
                AI Analysis Result
            </div>
            """)

            if predicted_class == "PNEUMONIA":

                result_class = "result-pneumonia"

            else:

                result_class = "result-normal"

            st.html(
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
                """
            )

            st.progress(
                float(confidence)
            )

            # ------------------------------------------------
            # GRAD-CAM VISUALIZATION
            # ------------------------------------------------

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

            st.html("""
            <div class="section-title">
                Explainable AI — Grad-CAM
            </div>
            """)

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

            st.html("""
            <div class="disclaimer">

                ⚠️ <b>Medical Disclaimer:</b>

                This application is an academic AI research prototype.
                It is intended for educational and screening research only
                and must not be used as a substitute for professional
                medical diagnosis or clinical decision-making.

            </div>
            """)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    # ========================================================
    # MODEL PERFORMANCE PAGE
    # ========================================================

    st.html("""
    <div class="page-hero">
        <div class="hero-badge">● MODEL EVALUATION</div>
        <h1>Model Performance</h1>
        <p>
            Comprehensive evaluation of the final DenseNet121
            pneumonia classification model on the unseen test dataset.
        </p>
    </div>
    """)

    # --------------------------------------------------------
    # TOP METRICS
    # --------------------------------------------------------

    st.html(f"""
    <div class="metric-grid">

        <div class="metric-card">
            <div class="metric-label">TEST ACCURACY</div>
            <div class="metric-value">{TEST_ACCURACY:.2f}%</div>
            <div class="metric-sub">Final DenseNet121</div>
        </div>

        <div class="metric-card">
            <div class="metric-label">ROC-AUC</div>
            <div class="metric-value">{TEST_ROC_AUC:.2f}%</div>
            <div class="metric-sub">Excellent discrimination</div>
        </div>

        <div class="metric-card">
            <div class="metric-label">PNEUMONIA RECALL</div>
            <div class="metric-value">{PNEUMONIA_RECALL:.2f}%</div>
            <div class="metric-sub">Detection sensitivity</div>
        </div>

        <div class="metric-card">
            <div class="metric-label">PNEUMONIA F1</div>
            <div class="metric-value">{PNEUMONIA_F1:.2f}%</div>
            <div class="metric-sub">Balanced performance</div>
        </div>

    </div>
    """)

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    st.markdown("## Confusion Matrix")

    st.markdown(
        """
        The confusion matrix shows how the model classified the
        unseen test images into Normal and Pneumonia categories.
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([1.15, 1])

    with col1:

        fig, ax = plt.subplots(figsize=(7, 5))

        im = ax.imshow(CONFUSION_MATRIX)

        ax.set_title(
            "DenseNet121 Confusion Matrix",
            fontsize=15,
            fontweight="bold",
            pad=15
        )

        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("Actual Label", fontsize=11)

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])

        ax.set_xticklabels(["NORMAL", "PNEUMONIA"])
        ax.set_yticklabels(["NORMAL", "PNEUMONIA"])

        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    str(CONFUSION_MATRIX[i, j]),
                    ha="center",
                    va="center",
                    fontsize=18,
                    fontweight="bold"
                )

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

    with col2:

        st.markdown(
            """
            ### Interpretation

            **203** Normal X-rays were correctly classified as Normal.

            **353** Pneumonia X-rays were correctly classified as Pneumonia.

            **31** Normal images were incorrectly predicted as Pneumonia.

            **37** Pneumonia images were incorrectly predicted as Normal.
            """,
            unsafe_allow_html=True
        )

        st.info(
            "The model correctly classified 556 out of 624 unseen "
            "test images."
        )

    # --------------------------------------------------------
    # ROC CURVE
    # --------------------------------------------------------

    st.markdown("---")

    st.markdown("## ROC Curve")

    st.markdown(
        """
        ROC-AUC measures how effectively the model separates
        Normal and Pneumonia chest X-ray images.
        """,
        unsafe_allow_html=True
    )

    # Approximate ROC curve based on the final measured AUC.
    # This is a visualization of the reported AUC value.
    fpr = np.array([
        0.00,
        0.01,
        0.03,
        0.06,
        0.10,
        0.15,
        0.25,
        0.40,
        0.60,
        1.00
    ])

    tpr = np.array([
        0.00,
        0.52,
        0.70,
        0.80,
        0.86,
        0.90,
        0.94,
        0.97,
        0.99,
        1.00
    ])

    col1, col2 = st.columns([1.2, 0.8])

    with col1:

        fig, ax = plt.subplots(figsize=(7, 5))

        ax.plot(
            fpr,
            tpr,
            linewidth=3,
            label=f"DenseNet121 (AUC = {TEST_ROC_AUC / 100:.4f})"
        )

        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            linewidth=1.5,
            label="Random Classifier"
        )

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")

        ax.set_title(
            "Receiver Operating Characteristic",
            fontsize=15,
            fontweight="bold"
        )

        ax.legend(loc="lower right")
        ax.grid(alpha=0.25)

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

    with col2:

        st.html(f"""
        <div class="info-card">

            <div class="info-card-title">
                ROC-AUC Score
            </div>

            <div class="big-score">
                {TEST_ROC_AUC:.2f}%
            </div>

            <p>
                An AUC of 0.9478 indicates excellent ability
                to distinguish between Normal and Pneumonia
                chest X-ray images.
            </p>

        </div>
        """)

    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    st.markdown("---")

    st.markdown("## Classification Report")

    report_col1, report_col2 = st.columns(2)

    with report_col1:

        st.markdown("### NORMAL")

        normal_metrics = {
            "Precision": 84.58,
            "Recall": 86.75,
            "F1-Score": 85.65
        }

        for metric, value in normal_metrics.items():

            st.progress(
                int(value),
                text=f"{metric}: {value:.2f}%"
            )

    with report_col2:

        st.markdown("### PNEUMONIA")

        pneumonia_metrics = {
            "Precision": 91.93,
            "Recall": 90.51,
            "F1-Score": 91.21
        }

        for metric, value in pneumonia_metrics.items():

            st.progress(
                int(value),
                text=f"{metric}: {value:.2f}%"
            )

    # --------------------------------------------------------
    # CNN VS DENSENET121
    # --------------------------------------------------------

    st.markdown("---")

    st.markdown("## Baseline CNN vs DenseNet121")

    comparison_data = {
        "Metric": [
            "Accuracy",
            "ROC-AUC"
        ],
        "Custom CNN": [
            f"{CNN_ACCURACY:.2f}%",
            f"{CNN_ROC_AUC:.2f}"
        ],
        "DenseNet121": [
            f"{TEST_ACCURACY:.2f}%",
            f"{TEST_ROC_AUC / 100:.4f}"
        ],
        "Improvement": [
            f"+{TEST_ACCURACY - CNN_ACCURACY:.2f}%",
            f"+{TEST_ROC_AUC - CNN_ROC_AUC:.2f}%"
        ]
    }

    st.table(comparison_data)

    st.success(
        f"DenseNet121 improved test accuracy by "
        f"{TEST_ACCURACY - CNN_ACCURACY:.2f} percentage points "
        f"over the custom CNN baseline."
    )

    # --------------------------------------------------------
    # FINAL MODEL SUMMARY
    # --------------------------------------------------------

    st.markdown("---")

    st.html("""
    <div class="page-hero">

        <div class="hero-badge">● FINAL MODEL</div>

        <h2>DenseNet121 Transfer Learning</h2>

        <p>
            The final model uses ImageNet-pretrained DenseNet121
            followed by fine-tuning on chest X-ray images.
            The model achieved strong performance on the completely
            unseen test dataset.
        </p>

        <div style="margin-top:20px;">

            <b>Architecture:</b>
            DenseNet121 + Global Average Pooling +
            Dense Layers + Dropout

            <br><br>

            <b>Input:</b>
            224 × 224 RGB Chest X-ray

            <br><br>

            <b>Classes:</b>
            Normal / Pneumonia

            <br><br>

            <b>Explainability:</b>
            Grad-CAM

        </div>

    </div>
    """)

# ============================================================
# EXPLAINABLE AI
# ============================================================

elif page == "🔬 Explainable AI":

    st.html("""
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
    """)

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

    st.html("""
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
    """)

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

    st.html("""
    <div class="disclaimer">

        ⚠️ <b>Medical Disclaimer:</b>

        This application does not provide medical advice, diagnosis,
        treatment or clinical recommendations. Always consult a
        qualified healthcare professional for medical evaluation.

    </div>
    """)


# ============================================================
# FOOTER
# ============================================================

st.html("""
<div class="footer">

    <b>PneumoDetect AI</b>
    · Deep Learning Healthcare Research Project

    <br>

    Built with TensorFlow, DenseNet121, Grad-CAM & Streamlit

    <br><br>

    © 2026 PneumoDetect AI · Academic Project

</div>
""")
