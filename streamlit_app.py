"""
Streamlit interactive app for Drywall QA Prompted Segmentation.

Upload a construction image, type a text prompt, and get instant
segmentation results with adjustable post-processing controls.

Usage:
    streamlit run streamlit_app.py
"""

import io
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, PostprocessConfig
from src.evaluate import load_trained_model
from src.model import build_model, load_processor
from src.process import postprocess_mask, clean_boundaries, create_overlay


# ─── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Drywall QA — Prompted Segmentation",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .main-header h1 {
        color: #e0e0e0;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
    }
    .main-header p {
        color: #90caf9;
        font-size: 1rem;
        margin-top: 0.3rem;
    }

    .metric-card {
        background: linear-gradient(135deg, #1e1e30 0%, #2a2a45 100%);
        border: 1px solid #3a3a5c;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .metric-card h3 {
        color: #80cbc4;
        font-size: 0.85rem;
        font-weight: 500;
        margin: 0 0 0.3rem 0;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-card p {
        color: #ffffff;
        font-size: 1.6rem;
        font-weight: 700;
        margin: 0;
    }

    .result-label {
        font-size: 0.9rem;
        font-weight: 600;
        color: #b0bec5;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.3rem;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    div[data-testid="stSidebar"] .stMarkdown p,
    div[data-testid="stSidebar"] .stMarkdown h1,
    div[data-testid="stSidebar"] .stMarkdown h2,
    div[data-testid="stSidebar"] .stMarkdown h3 {
        color: #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)


# ─── Helpers (PIL-based, no cv2) ──────────────────────────────

def _resize_array(arr: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize a 2D numpy array using PIL (avoids cv2 dependency)."""
    img = Image.fromarray(arr)
    resized = img.resize((width, height), Image.BILINEAR)
    return np.array(resized, dtype=arr.dtype)


def _apply_inferno_colormap(gray: np.ndarray) -> np.ndarray:
    """Apply inferno colormap to a uint8 grayscale image."""
    import matplotlib.cm as cm
    normed = gray.astype(np.float32) / 255.0
    colored = cm.inferno(normed)  # (H, W, 4) RGBA float
    return (colored[:, :, :3] * 255).astype(np.uint8)


def _image_to_png_bytes(arr: np.ndarray) -> bytes:
    """Convert a numpy image array to PNG bytes via PIL."""
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ─── Model Loading (cached) ──────────────────────────────────
@st.cache_resource(show_spinner="Loading CLIPSeg model...")
def load_model_and_processor():
    """Load config, model, and processor. Cached across reruns."""
    config = load_config(str(PROJECT_ROOT / "configs" / "default.yaml"))

    processor = load_processor(config)

    checkpoint_path = PROJECT_ROOT / "outputs" / "best_model.pt"
    if checkpoint_path.exists():
        model = load_trained_model(str(checkpoint_path), config)
        model_source = "Fine-tuned checkpoint (local `best_model.pt`)"
    else:
        try:
            from huggingface_hub import hf_hub_download
            hf_path = hf_hub_download(
                repo_id="Deepak-TP/drywall-qa-prompted-segmentation", 
                filename="best_model.pt"
            )
            model = load_trained_model(hf_path, config)
            model_source = "Fine-tuned checkpoint (Hugging Face `Deepak-TP`)"
        except Exception as e:
            model = build_model(config)
            model_source = "Pre-trained CLIPSeg (fallback base model)"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    return model, processor, config, device, model_source


@torch.no_grad()
def run_segmentation(
    image: Image.Image,
    prompt: str,
    model,
    processor,
    config,
    device,
    threshold: float,
    morph_kernel: int,
    min_area: int,
):
    """Run prompted segmentation on a PIL image."""
    original_size = image.size  # (W, H)

    inputs = processor(
        text=[prompt],
        images=[image],
        return_tensors="pt",
        padding="max_length",
        max_length=77,
        truncation=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model(**inputs)
    logits = outputs.logits.cpu()

    # Squeeze ALL extra dims to get a clean (H, W) tensor
    while logits.dim() > 2:
        logits = logits.squeeze(0)

    image_np = np.array(image)

    # Raw probability map — resize using PIL
    raw_probs = torch.sigmoid(logits).numpy().astype(np.float32)
    raw_probs_resized = _resize_array(raw_probs, original_size[0], original_size[1])

    # Resize logits for post-processing
    logits_np = logits.numpy().astype(np.float32)
    logits_resized = _resize_array(logits_np, original_size[0], original_size[1])

    # Post-process with user-adjusted params
    pp_config = PostprocessConfig(
        threshold=threshold,
        morph_kernel_size=morph_kernel,
        min_component_area=min_area,
        use_opening=True,
        use_closing=True,
    )
    processed_mask = postprocess_mask(logits_resized, pp_config)
    processed_mask = clean_boundaries(processed_mask, kernel_size=3)

    # Overlay
    overlay = create_overlay(image_np, processed_mask, color=(0, 255, 0), alpha=0.4)

    # Stats
    mask_pixels = (processed_mask > 0).sum()
    total_pixels = processed_mask.shape[0] * processed_mask.shape[1]
    coverage = mask_pixels / total_pixels * 100
    confidence = float(raw_probs_resized[processed_mask > 0].mean()) if mask_pixels > 0 else 0.0

    return {
        "image": image_np,
        "raw_heatmap": raw_probs_resized,
        "mask": processed_mask,
        "overlay": overlay,
        "coverage": coverage,
        "confidence": confidence,
        "mask_pixels": int(mask_pixels),
    }


# ─── Main App ────────────────────────────────────────────────
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏗️ Drywall QA — Prompted Segmentation</h1>
        <p>Upload a construction image • Describe the defect • Get instant segmentation</p>
    </div>
    """, unsafe_allow_html=True)

    # Load model
    model, processor, config, device, model_source = load_model_and_processor()

    # ─── Sidebar ──────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Settings")

        st.markdown("---")
        st.markdown("### 📝 Text Prompt")

        prompt_presets = {
            "Custom...": "",
            "🔨 Wall crack": "wall crack",
            "🧱 Structural crack": "structural crack",
            "📐 Drywall joint": "drywall joint",
            "🪡 Taping area": "taping area",
            "🔗 Drywall seam": "drywall seam",
            "💥 Surface crack": "surface crack",
            "🏗️ Concrete crack": "concrete crack",
        }
        preset = st.selectbox("Quick prompts", list(prompt_presets.keys()))

        if preset == "Custom...":
            prompt = st.text_input(
                "Enter prompt",
                placeholder="e.g. crack on wall",
            )
        else:
            prompt = prompt_presets[preset]
            st.info(f'Using: **"{prompt}"**')

        st.markdown("---")
        st.markdown("### 🎛️ Post-Processing")

        threshold = st.slider(
            "Confidence threshold",
            min_value=0.1, max_value=0.9, value=0.5, step=0.05,
            help="Higher = stricter detection (fewer false positives)",
        )
        morph_kernel = st.slider(
            "Morphology kernel size",
            min_value=3, max_value=15, value=5, step=2,
            help="Larger = smoother masks (removes small noise)",
        )
        min_area = st.slider(
            "Min component area (px)",
            min_value=0, max_value=500, value=100, step=25,
            help="Remove detected regions smaller than this",
        )

        st.markdown("---")
        st.markdown("### 📊 Model Info")
        st.caption(f"**Source:** {model_source}")
        st.caption(f"**Device:** `{device}`")
        st.caption(f"**Model:** `{config.model.name}`")

    # ─── Image Upload ─────────────────────────────────────
    uploaded = st.file_uploader(
        "Upload a construction / drywall image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="Supports JPG, PNG, BMP, and WebP formats",
    )

    if uploaded is not None and prompt:
        image = Image.open(uploaded).convert("RGB")

        with st.spinner(f'🔍 Segmenting with prompt: **"{prompt}"**...'):
            results = run_segmentation(
                image, prompt, model, processor, config, device,
                threshold, morph_kernel, min_area,
            )

        # ─── Metrics Row ──────────────────────────────────
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Coverage</h3>
                <p>{results['coverage']:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Avg Confidence</h3>
                <p>{results['confidence']:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Mask Pixels</h3>
                <p>{results['mask_pixels']:,}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("")

        # ─── Results Grid ─────────────────────────────────
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown('<p class="result-label">📷 Original Image</p>', unsafe_allow_html=True)
            st.image(results["image"], use_container_width=True)

        with col_b:
            st.markdown('<p class="result-label">🎯 Overlay</p>', unsafe_allow_html=True)
            st.image(results["overlay"], use_container_width=True)

        col_c, col_d = st.columns(2)

        with col_c:
            st.markdown('<p class="result-label">🔥 Confidence Heatmap</p>', unsafe_allow_html=True)
            heatmap_rgb = _apply_inferno_colormap(
                (results["raw_heatmap"] * 255).astype(np.uint8)
            )
            st.image(heatmap_rgb, use_container_width=True)

        with col_d:
            st.markdown('<p class="result-label">🖼️ Binary Mask</p>', unsafe_allow_html=True)
            st.image(results["mask"], use_container_width=True)

        # ─── Download Buttons ─────────────────────────────
        st.markdown("---")
        st.markdown("### 📥 Download Results")

        dl_col1, dl_col2, dl_col3 = st.columns(3)

        with dl_col1:
            mask_bytes = _image_to_png_bytes(results["mask"])
            st.download_button(
                "⬇️ Binary Mask (PNG)",
                data=mask_bytes,
                file_name=f"mask_{prompt.replace(' ', '_')}.png",
                mime="image/png",
            )

        with dl_col2:
            overlay_bytes = _image_to_png_bytes(results["overlay"])
            st.download_button(
                "⬇️ Overlay (PNG)",
                data=overlay_bytes,
                file_name=f"overlay_{prompt.replace(' ', '_')}.png",
                mime="image/png",
            )

        with dl_col3:
            heatmap_bytes = _image_to_png_bytes(heatmap_rgb)
            st.download_button(
                "⬇️ Heatmap (PNG)",
                data=heatmap_bytes,
                file_name=f"heatmap_{prompt.replace(' ', '_')}.png",
                mime="image/png",
            )

    elif uploaded is not None and not prompt:
        st.warning("⬅️ Please enter a text prompt in the sidebar to start segmentation.")
    else:
        st.info("👆 Upload an image above and select a prompt to begin.")

        # Show example usage
        with st.expander("📖 How to use this app"):
            st.markdown("""
            1. **Upload** a construction or drywall image (JPG, PNG)
            2. **Select a prompt** from the presets or type a custom one
            3. **Adjust** post-processing settings in the sidebar
            4. **View** the segmentation results in a 4-panel grid
            5. **Download** the mask, overlay, or heatmap

            **Example prompts:**
            - `"wall crack"` — detects cracks on walls
            - `"drywall joint"` — highlights drywall seams
            - `"taping area"` — finds areas that need tape application
            - `"structural crack"` — identifies deep structural damage
            """)


if __name__ == "__main__":
    main()
