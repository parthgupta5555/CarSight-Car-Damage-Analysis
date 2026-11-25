import io

import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image, UnidentifiedImageError

from model_helper import (
    CLASS_NAMES,
    IMAGE_TRANSFORM,
    load_model,
    predict_image,
)



st.set_page_config(
    page_title="CarSight | Intelligent Car Damage Analysis",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp {
        background:
            radial-gradient(circle at 85% 2%, rgba(31,111,235,.12), transparent 26rem),
            linear-gradient(180deg, #0b1220 0%, #0d1524 48%, #0a101b 100%);
        color: #edf3ff;
    }
    [data-testid="stHeader"] { background: rgba(11,18,32,.75); }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #101b2d 0%, #0d1625 100%);
        border-right: 1px solid rgba(148,163,184,.14);
    }
    [data-testid="stSidebar"] * { color: #dce7f8; }
    .block-container { padding-top: 2.1rem; padding-bottom: 3rem; max-width: 1450px; }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -.035em; }
    .eyebrow {
        color: #78a9ff; font-size: .76rem; font-weight: 700;
        letter-spacing: .16em; text-transform: uppercase; margin-bottom: .55rem;
    }
    .hero-title {
        font-family: 'Space Grotesk', sans-serif; font-size: clamp(2.1rem, 4vw, 3.45rem);
        line-height: 1.05; font-weight: 700; letter-spacing: -.055em;
        color: #f5f8ff; margin: 0;
    }
    .hero-copy { color: #9eacc2; font-size: 1.02rem; max-width: 760px; margin-top: .85rem; }
    .panel {
        background: linear-gradient(145deg, rgba(24,39,61,.88), rgba(15,26,43,.92));
        border: 1px solid rgba(148,163,184,.16);
        border-radius: 18px; padding: 1.15rem 1.25rem;
        box-shadow: 0 14px 40px rgba(0,0,0,.16);
    }
    .panel-label { color: #91a4bf; font-size: .76rem; text-transform: uppercase; letter-spacing: .12em; font-weight: 700; }
    .metric-value { color: #f3f7ff; font-family: 'Space Grotesk', sans-serif; font-size: 1.65rem; font-weight: 700; margin-top: .35rem; }
    .muted, .small-note { color: #9eacc2; }
    .small-note { font-size: .8rem; }
    .result-title { font-family: 'Space Grotesk', sans-serif; color: #f5f8ff; font-size: 1.8rem; font-weight: 700; line-height: 1.15; }
    .status-pill {
        display: inline-block; padding: .32rem .7rem; border-radius: 999px;
        background: rgba(52,211,153,.12); border: 1px solid rgba(52,211,153,.3);
        color: #6ee7b7; font-size: .75rem; font-weight: 700; letter-spacing: .06em;
    }
    div[data-testid="stFileUploader"] {
        background: rgba(19,32,51,.7); border: 1px dashed rgba(120,169,255,.45);
        border-radius: 15px; padding: .65rem;
    }
    div.stButton > button {
        width: 100%; border-radius: 11px; border: 1px solid #3b82f6;
        background: linear-gradient(90deg, #2563eb, #397cf6);
        color: white; font-weight: 700; padding: .72rem 1rem;
    }
    div.stButton > button:hover {
        border-color: #8ab5ff; background: linear-gradient(90deg, #1d4ed8, #2563eb);
    }
    div[data-testid="stImage"] img { border-radius: 14px; }
    hr { border-color: rgba(148,163,184,.16); }
    #MainMenu, footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# Model loading

@st.cache_resource(show_spinner="Initializing CarSight...")
def get_model():
    # The helper loads the checkpoint and selects CUDA when available.
    return load_model("saved_model.pth")


# Grad-CAM

def create_gradcam(pil_image, model, device, target_class_index):
    """
    Create a Grad-CAM overlay for one selected output class.
    Gradients must remain enabled for this operation.
    """
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        from pytorch_grad_cam.utils.image import show_cam_on_image
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "Grad-CAM package is missing. Install it with: "
            "pip install grad-cam"
        ) from exc

    rgb_image = pil_image.convert("RGB").resize((224, 224))
    rgb_array = np.asarray(rgb_image, dtype=np.float32) / 255.0
    input_tensor = IMAGE_TRANSFORM(pil_image.convert("RGB")).unsqueeze(0).to(device)

    target_layers = [model.model.layer4[-1]]
    targets = [ClassifierOutputTarget(int(target_class_index))]

    # GradCAM internally computes gradients; do not use no_grad/inference_mode here.
    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    overlay = show_cam_on_image(
        rgb_array,
        grayscale_cam,
        use_rgb=True,
        image_weight=0.55,
    )
    return Image.fromarray(overlay)


# Sidebar
with st.sidebar:
    st.markdown("## 🚘 CarSight")
    st.caption("SEE THE DAMAGE. UNDERSTAND THE IMPACT.")
    st.markdown("---")
    st.markdown("### About CarSight")
    st.write(
        "CarSight analyzes car images, classifies visible front or rear damage, "
        "compares category scores, and provides a visual explanation of the model's focus."
    )
    with st.expander("Technical details"):
        st.markdown(
            """
            - **Task:** Six-class car-damage image classification
            - **Framework:** PyTorch
            - **Deployed architecture:** ResNet50
            - **Approach:** Transfer learning with layer 4 fine-tuning
            - **Input size:** 224 × 224
            - **Explainability:** Grad-CAM
            """
        )
    st.markdown("---")
    st.markdown("### Supported car-damage categories")
    for category in CLASS_NAMES:
        st.markdown(f"• {category}")
    st.markdown("---")
    st.caption("Computer Vision • Deep Learning • Image Classification")


# Main page header
st.markdown(
    '<div class="eyebrow">COMPUTER VISION · DEEP LEARNING</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="hero-title">Intelligent Car<br>Damage Analysis</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="hero-copy">Explore CarSight’s car-damage analysis workflow: '
    'classify visible car condition, compare prediction scores across six categories, '
    'and inspect the image regions influencing a selected prediction.</div>',
    unsafe_allow_html=True,
)
st.write("")

m1, m2, m3 = st.columns(3)
for col, label, value, detail in [
    (m1, "DAMAGE CLASSIFICATION", "6 categories", "Front and rear car conditions"),
    (m2, "PREDICTION ANALYSIS", "Class-wise scores", "Compare all six model outputs"),
    (m3, "VISUAL EXPLAINABILITY", "Grad-CAM", "Inspect regions influencing a prediction"),
]:
    with col:
        st.markdown(
            f'<div class="panel"><div class="panel-label">{label}</div>'
            f'<div class="metric-value">{value}</div>'
            f'<div class="small-note">{detail}</div></div>',
            unsafe_allow_html=True,
        )

st.write("")


# Upload and prediction
left, right = st.columns([1.02, 0.98], gap="large")

with left:
    st.markdown("### 01 · Upload car image")
    st.markdown(
        '<div class="muted">Choose a clear image showing the front or rear of a car.</div>',
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        "Upload a car image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG and PNG.",
    )

    image = None
    if uploaded_file is not None:
        try:
            image = Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB")
            st.image(
                image,
                caption=f"Uploaded image · {uploaded_file.name}",
                use_container_width=True,
            )
        except (UnidentifiedImageError, OSError):
            st.error("This image could not be opened. Please upload a valid JPG or PNG.")
            image = None

    analyze_clicked = st.button("✦  Analyze car image", disabled=image is None)

    if st.button("↻  Reset analysis"):
        st.session_state.pop("analysis_result", None)
        st.session_state.pop("analysis_image", None)
        st.session_state.pop("analysis_device", None)
        st.session_state.pop("gradcam_image", None)
        st.rerun()

    if analyze_clicked and image is not None:
        try:
            with st.spinner("Analyzing your car image..."):
                model, device = get_model()
                result = predict_image(image, model, device)

            st.session_state["analysis_result"] = result
            st.session_state["analysis_image"] = image.copy()
            st.session_state["analysis_device"] = device
            st.session_state.pop("gradcam_image", None)
        except FileNotFoundError:
            st.error(
                "Could not find `saved_model.pth`. Place the trained checkpoint "
                "in the same folder as this Streamlit file."
            )
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.caption("Check that the checkpoint matches the ResNet50 model architecture.")


# Results
with right:
    st.markdown("### 02 · Analysis results")
    if "analysis_result" not in st.session_state:
        st.markdown(
            """
            <div class="panel" style="min-height:250px; display:flex; flex-direction:column;
                 align-items:center; justify-content:center; text-align:center;">
                <div style="font-size:2.5rem;">✦</div>
                <div style="font-size:1.05rem; font-weight:700; color:#eaf1ff; margin-top:.6rem;">
                    Ready when you are
                </div>
                <div class="small-note" style="margin-top:.35rem;">
                    Upload a car image and select Analyze to see the model output
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        result = st.session_state["analysis_result"]
        device = st.session_state["analysis_device"]
        analyzed_image = st.session_state["analysis_image"]
        predicted_class = result["class_name"]
        confidence = result["confidence"]
        probabilities = result["probabilities"]

        st.markdown(
            '<div class="panel"><div class="panel-label">PREDICTED CAR-DAMAGE CATEGORY</div>'
            f'<div class="result-title" style="margin-top:.55rem;">{predicted_class}</div>'
            '<div style="margin-top:.8rem;"><span class="status-pill">ANALYSIS COMPLETE</span></div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.write("")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f'<div class="panel"><div class="panel-label">TOP PREDICTION SCORE</div>'
                f'<div class="metric-value">{confidence:.1%}</div>'
                '<div class="small-note">Softmax score for the predicted class</div></div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f'<div class="panel"><div class="panel-label">INFERENCE DEVICE</div>'
                f'<div class="metric-value">{str(device).upper()}</div>'
                '<div class="small-note">Device used for this prediction</div></div>',
                unsafe_allow_html=True,
            )

        st.write("")
        st.markdown("#### Class-wise prediction scores")
        chart_df = pd.DataFrame({
            "Car-damage category": CLASS_NAMES,
            "Score": [probabilities[name] for name in CLASS_NAMES],
        }).sort_values("Score", ascending=True)

        fig = px.bar(
            chart_df,
            x="Score",
            y="Car-damage category",
            orientation="h",
            text=chart_df["Score"].map(lambda score: f"{score:.1%}"),
            range_x=[0, 1],
        )
        fig.update_traces(
            marker_color="#4f8df7",
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}<br>Score: %{x:.2%}<extra></extra>",
        )
        fig.update_layout(
            height=340,
            margin=dict(l=5, r=45, t=12, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#dce7f8", family="DM Sans"),
            xaxis=dict(
                title=None, tickformat=".0%",
                gridcolor="rgba(148,163,184,.13)", zeroline=False,
            ),
            yaxis=dict(title=None, gridcolor="rgba(0,0,0,0)"),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.caption(
            "Scores are softmax outputs. They are not necessarily calibrated probabilities "
            "or a guarantee of real-world accuracy."
        )

        with st.expander("View all prediction scores"):
            score_df = pd.DataFrame([
                {"Car-damage category": name, "Score": f"{probabilities[name]:.2%}"}
                for name in CLASS_NAMES
            ]).sort_values("Score", ascending=False)
            st.dataframe(score_df, hide_index=True, use_container_width=True)



# Full-width visual explainability section
if "analysis_result" in st.session_state:
    st.markdown("---")
    with st.container():
        st.markdown('<div class="eyebrow">MODEL INTERPRETABILITY</div>', unsafe_allow_html=True)
        # Grad-CAM
        st.markdown("---")
        st.markdown("### 03 · Visual explanation with Grad-CAM")
        st.write(
            "Explore which image regions influence a selected car-damage category. "
            "The explanation is displayed across the page rather than in a narrow results column."
        )

        selected_class = st.selectbox(
            "Class to visualize",
            options=CLASS_NAMES,
            index=CLASS_NAMES.index(predicted_class),
        )
        if st.button("Generate Grad-CAM"):
            try:
                with st.spinner(f"Generating Grad-CAM for {selected_class}..."):
                    model, device = get_model()
                    selected_index = CLASS_NAMES.index(selected_class)
                    cam_image = create_gradcam(
                        analyzed_image, model, device, selected_index
                    )
                    st.session_state["gradcam_image"] = cam_image
                    st.session_state["gradcam_class"] = selected_class
            except ImportError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Could not generate Grad-CAM: {exc}")

        if "gradcam_image" in st.session_state:
            cam_col1, cam_col2 = st.columns([1, 1], gap="large")
            with cam_col1:
                st.image(analyzed_image, caption="Original car image", use_container_width=True)
            with cam_col2:
                st.image(
                    st.session_state["gradcam_image"],
                    caption=f'Grad-CAM · {st.session_state.get("gradcam_class", selected_class)}',
                    use_container_width=True,
                )
            st.caption(
                "Interpretation aid only: Grad-CAM is not a pixel-level damage mask "
                "and does not establish the exact physical extent of damage."
            )



# Footer
st.markdown("---")
st.markdown(
    '<div style="text-align:center; color:#71839e; font-size:.78rem;">'
    'CARSIGHT · INTELLIGENT CAR DAMAGE ANALYSIS · COMPUTER VISION'
    '</div>',
    unsafe_allow_html=True,
)
