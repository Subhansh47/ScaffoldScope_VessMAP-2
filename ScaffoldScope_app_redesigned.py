from pathlib import Path
import io
import csv

import numpy as np
import streamlit as st
import torch
from PIL import Image
from scaffoldscope.model import UNet
from scaffoldscope.analysis import analyze

ROOT = Path(__file__).resolve().parent
CKPT = ROOT / "models" / "scaffoldscope_unet.pt"

st.set_page_config(
    page_title="ScaffoldScope | Vascular Analysis",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Visual design ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --bg:#06111f; --panel:#0b1929; --panel2:#0d2033; --line:#1b344b; --cyan:#24d9e8; --blue:#39aef5; --muted:#9bb2ca; --text:#edf6ff; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background: radial-gradient(ellipse at 80% 0%, #102641 0%, #071321 42%, #050d18 100%); color:var(--text); }
[data-testid="stHeader"] { background:rgba(5,13,24,.7); }
.block-container { padding-top:1.5rem; padding-bottom:2rem; max-width:1600px; }
h1,h2,h3 { font-family:'Manrope',sans-serif !important; color:var(--text) !important; letter-spacing:-.03em; }
h1 { font-size:2.25rem !important; font-weight:800 !important; }
p, label, .stMarkdown { color:#dce9f8; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#0a1727 0%,#07111e 100%); border-right:1px solid #193047; }
[data-testid="stSidebar"] > div { padding-top:1.3rem; }
.brand { font-family:'Manrope',sans-serif; font-weight:800; font-size:1.55rem; letter-spacing:-.06em; color:#f2f8ff; margin-bottom:0; }
.brand span { color:var(--cyan); }
.brand-sub { font-size:.68rem; color:#9db5d0; letter-spacing:.18em; text-transform:uppercase; margin-bottom:1.8rem; }
.hero-sub { color:#9eb8d5; font-size:1.02rem; margin-top:-.6rem; }
.eyebrow { color:var(--cyan); font-size:.72rem; font-weight:700; letter-spacing:.15em; text-transform:uppercase; }
.panel { background:linear-gradient(145deg,rgba(14,31,50,.96),rgba(8,20,34,.98)); border:1px solid #203950; border-radius:16px; padding:1rem 1.1rem; box-shadow:0 12px 35px rgba(0,0,0,.12); }
.panel-title { color:#eef7ff; font-size:1rem; font-weight:700; margin-bottom:.85rem; }
.metric-card { border:1px solid #21445a; background:linear-gradient(145deg,#0e2638,#0a1929); border-radius:14px; padding:1rem; min-height:128px; }
.metric-label { font-size:.79rem; color:#a9bfd6; margin-bottom:.55rem; }
.metric-value { font-family:'Manrope',sans-serif; font-size:1.65rem; font-weight:800; color:#f3f9ff; letter-spacing:-.04em; }
.metric-foot { font-size:.72rem; color:#89a5c1; margin-top:.3rem; }
.cyan { border-color:#087e94; } .green { border-color:#087d68; } .coral { border-color:#994353; } .purple { border-color:#6742a0; }
.status { display:inline-block; padding:.32rem .62rem; border-radius:999px; color:#58e6c0; background:#10372f; border:1px solid #1d6758; font-size:.75rem; font-weight:700; }
.drop-note { border:1px dashed #148ca8; background:rgba(7,36,54,.65); border-radius:14px; padding:1.15rem; color:#9fc5dc; }
.stButton > button, .stDownloadButton > button { border-radius:10px; border:1px solid #18b7cc; background:linear-gradient(135deg,#1dd2df,#1aaac6); color:#031521; font-weight:700; min-height:2.6rem; }
.stButton > button:hover, .stDownloadButton > button:hover { border-color:#7af5ff; color:#031521; }
[data-testid="stFileUploader"] { background:rgba(7,29,45,.72); border:1px dashed #1aa5bf; border-radius:13px; padding:.5rem; }
[data-testid="stMetric"] { background:#0b1c2d; border:1px solid #1c3a50; padding:1rem; border-radius:12px; }
[data-testid="stMetricLabel"] { color:#a8c1d9 !important; }
[data-testid="stMetricValue"] { color:#f3f8ff !important; }
hr { border-color:#20374b; }
footer { visibility:hidden; }
.small-muted { color:#8ea8c2; font-size:.78rem; }
</style>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown('<div style="font-size:2rem;margin-bottom:.3rem">✳️</div><div class="brand">Scaffold<span>Scope</span></div><div class="brand-sub">Vascular image analysis</div>', unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio(
        "WORKSPACE",
        ["Overview", "Analyze Image", "Batch Analysis", "Results & Reports", "Model Status"],
        index=0,
        label_visibility="visible",
    )
    st.markdown("---")
    if CKPT.exists():
        st.markdown('<span class="status">● Model checkpoint found</span>', unsafe_allow_html=True)
        st.caption("ScaffoldScope U-Net checkpoint detected.")
    else:
        st.warning("Model not trained yet")
        st.caption("Train the U-Net to enable model-based predictions.")
    st.markdown("---")
    st.caption("RESEARCH WORKSPACE")
    st.markdown('<div class="small-muted">Estimates only · Validate with calibrated reference measurements.</div>', unsafe_allow_html=True)

# ---------- Model ----------
@st.cache_resource
def load_model():
    if not CKPT.exists():
        return None, None
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CKPT, map_location=device, weights_only=False)
    model = UNet(base=checkpoint.get("base_channels", 32)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, (device, checkpoint.get("image_size", 256))

model, info = load_model()

# ---------- Header ----------
left, right = st.columns([3, 1])
with left:
    st.markdown('<div class="eyebrow">MICROSCOPY INTELLIGENCE WORKSPACE</div>', unsafe_allow_html=True)
    st.title("Vascular Analysis")
    st.markdown('<div class="hero-sub">Quantify vessel architecture with a clear, reproducible workflow.</div>', unsafe_allow_html=True)
with right:
    st.markdown("<div style='text-align:right;padding-top:1rem'><span class='status'>● Local workspace</span></div>", unsafe_allow_html=True)

if page == "Model Status":
    st.subheader("Model status")
    if model is None:
        st.warning("No trained checkpoint found at models/scaffoldscope_unet.pt.")
        st.code("./train_mac.sh 20", language="bash")
    else:
        st.success("Model checkpoint loaded.")
        st.write(f"Device: `{info[0]}`")
        st.write(f"Model input size: `{info[1]} × {info[1]}`")
    st.stop()

if page == "Overview":
    st.markdown("""
    <div class="panel">
      <div class="panel-title">Welcome to ScaffoldScope</div>
      Upload a microscopy image to generate a vessel-mask prediction and inspect morphology measurements.
      <div class="small-muted" style="margin-top:.5rem">Supported formats: PNG, JPG, TIFF, BMP. Model-based analysis requires a trained checkpoint.</div>
    </div>
    """, unsafe_allow_html=True)
    st.write("")
elif page == "Batch Analysis":
    st.info("Batch analysis is a planned workspace. For now, analyze images one at a time so each prediction and its measurements can be reviewed.")
elif page == "Results & Reports":
    st.info("Upload and analyze an image below to create results. Downloadable CSV and mask exports will appear after analysis.")
elif page == "Analyze Image":
    st.markdown('<div class="eyebrow">NEW ANALYSIS</div>', unsafe_allow_html=True)

st.markdown('<div class="panel-title">01  /  Select microscopy image</div>', unsafe_allow_html=True)
uploaded_files = st.file_uploader(
    "Drop a microscopy image here or browse files",
    type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
    accept_multiple_files=(page == "Batch Analysis"),
    help="Use a real microscopy image. For physical measurements, enter a known pixel calibration.",
)
if uploaded_files and not isinstance(uploaded_files, list):
    uploaded_files = [uploaded_files]

if not uploaded_files:
    st.markdown('<div class="drop-note">↥ &nbsp; <b>Ready for your image</b><br><span class="small-muted">Your image stays in this local Streamlit session.</span></div>', unsafe_allow_html=True)
    if model is None:
        st.warning("No trained model checkpoint found. The UI is ready, but model-based segmentation requires training first.")
        st.code("./train_mac.sh 20", language="bash")
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="panel"><div class="eyebrow">STEP 01</div><div class="panel-title">Upload</div><div class="small-muted">Choose a microscopy image in a supported format.</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="panel"><div class="eyebrow">STEP 02</div><div class="panel-title">Segment</div><div class="small-muted">Run your trained U-Net to estimate vessel pixels.</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="panel"><div class="eyebrow">STEP 03</div><div class="panel-title">Measure & export</div><div class="small-muted">Review approximate morphology statistics and export outputs.</div></div>', unsafe_allow_html=True)
    st.stop()

threshold = st.sidebar.slider("Segmentation threshold", 0.05, 0.95, 0.50, 0.05)
min_obj = st.sidebar.slider("Remove objects smaller than (pixels)", 1, 100, 8)
px = st.sidebar.number_input("Pixel size (physical units / pixel)", min_value=0.0, value=0.0, step=0.1, help="Leave at 0 if calibration is unknown.")

for file in uploaded_files:
    try:
        original = Image.open(file).convert("L")
        arr = np.asarray(original)
    except Exception as exc:
        st.error(f"Could not read {file.name}: {exc}")
        continue

    st.markdown(f'<div class="panel-title">02  /  Image review <span class="small-muted">· {file.name}</span></div>', unsafe_allow_html=True)
    if model is None:
        st.image(original, caption="Input microscopy image", use_container_width=True)
        st.warning("A trained model checkpoint is required for vessel segmentation. No prediction or measurements have been fabricated.")
        continue

    device, size = info
    resized = original.resize((size, size), Image.Resampling.BILINEAR)
    x = np.asarray(resized, dtype=np.float32) / 255.0
    with torch.no_grad():
        prob = torch.sigmoid(model(torch.from_numpy(x[None, None]).to(device)))[0, 0].detach().cpu().numpy()

    prob_img = Image.fromarray(np.uint8(np.clip(prob, 0, 1) * 255)).resize(original.size, Image.Resampling.BILINEAR)
    prob_full = np.asarray(prob_img, dtype=np.float32) / 255.0
    stats = analyze(prob_full >= threshold, px if px > 0 else None, min_obj)

    original_rgb = np.stack([arr, arr, arr], axis=-1).astype(np.float32)
    overlay = original_rgb.copy()
    mask = stats["mask"].astype(bool)
    overlay[mask] = 0.50 * overlay[mask] + 0.50 * np.array([20, 220, 235], dtype=np.float32)

    image_col, metric_col = st.columns([1.25, 1])
    with image_col:
        st.markdown('<div class="panel-title">Image preview</div>', unsafe_allow_html=True)
        tab_original, tab_overlay, tab_mask = st.tabs(["Original", "Segmentation overlay", "Binary mask"])
        with tab_original:
            st.image(original, use_container_width=True)
        with tab_overlay:
            st.image(np.uint8(np.clip(overlay, 0, 255)), use_container_width=True)
        with tab_mask:
            st.image(np.uint8(mask * 255), use_container_width=True)
    with metric_col:
        st.markdown('<div class="panel-title">Analysis summary</div>', unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        with m1:
            st.markdown(f'<div class="metric-card green"><div class="metric-label">Vessel area</div><div class="metric-value">{stats["area_pixels"]:,}</div><div class="metric-foot">foreground pixels</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-card cyan"><div class="metric-label">Connected components</div><div class="metric-value">{stats["components"]:,}</div><div class="metric-foot">detected regions</div></div>', unsafe_allow_html=True)
        st.write("")
        m3, m4 = st.columns(2)
        with m3:
            st.markdown(f'<div class="metric-card coral"><div class="metric-label">Skeleton endpoints</div><div class="metric-value">{stats["endpoints"]:,}</div><div class="metric-foot">estimated endpoints</div></div>', unsafe_allow_html=True)
        with m4:
            st.markdown(f'<div class="metric-card purple"><div class="metric-label">Skeleton length</div><div class="metric-value">{stats["approx_length_pixels"]:,.0f}</div><div class="metric-foot">approximate pixels</div></div>', unsafe_allow_html=True)

    st.write("")
    chart_col, export_col = st.columns([1.2, 1])
    with chart_col:
        st.markdown('<div class="panel-title">Vessel morphology</div>', unsafe_allow_html=True)
        st.caption("Foreground profile across image rows. This is a visualization aid, not a diameter distribution.")
        row_profile = mask.mean(axis=1)
        st.line_chart({"Vessel pixel fraction by row": row_profile}, height=220)
    with export_col:
        st.markdown('<div class="panel-title">Export results</div>', unsafe_allow_html=True)
        st.caption("Export this image's computed measurements and binary mask.")
        csv_buffer = io.StringIO()
        writer = csv.writer(csv_buffer)
        writer.writerow(["filename", "threshold", "min_object_pixels", "pixel_size_physical_units", "area_pixels", "components", "endpoints", "approx_length_pixels", "area_scaled", "approx_length_scaled"])
        writer.writerow([file.name, threshold, min_obj, px if px > 0 else "", stats.get("area_pixels"), stats.get("components"), stats.get("endpoints"), stats.get("approx_length_pixels"), stats.get("area_scaled", ""), stats.get("approx_length_scaled", "")])
        st.download_button("↓  Download measurements (CSV)", csv_buffer.getvalue(), file_name=f"{Path(file.name).stem}_measurements.csv", mime="text/csv", use_container_width=True)
        mask_buffer = io.BytesIO()
        Image.fromarray(np.uint8(mask * 255)).save(mask_buffer, format="PNG")
        st.download_button("↓  Download segmentation mask (PNG)", mask_buffer.getvalue(), file_name=f"{Path(file.name).stem}_mask.png", mime="image/png", use_container_width=True)
        if px <= 0:
            st.info("Physical units are disabled because pixel calibration is unknown.")
    st.markdown('<div class="small-muted">Research-use estimates only. Skeleton length is approximate and measurements should be validated against expert annotations and calibrated reference data.</div>', unsafe_allow_html=True)
    st.divider()
