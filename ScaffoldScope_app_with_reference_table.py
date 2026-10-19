from pathlib import Path
import io
import csv

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from scaffoldscope.model import UNet
from scaffoldscope.analysis import analyze

ROOT = Path(__file__).resolve().parent
CKPT = ROOT / "models" / "scaffoldscope_unet.pt"

# Calibration supplied by the user: 1.1 pixels per micrometer.
PIXELS_PER_UM_DEFAULT = 1.1
UM_PER_PIXEL_DEFAULT = 1.0 / PIXELS_PER_UM_DEFAULT

# These are user-supplied reference-group values, not predictions from the uploaded image.
REFERENCE_ROWS = [
    {"Control": "36.6 ± 6.9", "MP": "58.8 ± 7.2", "p Value": ".02", "Measure": "Branch count"},
    {"Control": "15.8 ± 3.7", "MP": "27.1 ± 4.2", "p Value": ".03", "Measure": "Loop count"},
    {"Control": "9946.8 ± 1398.4", "MP": "13527.9 ± 117.6", "p Value": ".04", "Measure": "Total tube length (μm)"},
    {"Control": "24.1 ± 0.9", "MP": "29.0 ± 0.8", "p Value": ".00", "Measure": "Average vessel diameter (μm)"},
    {"Control": "9.3 ± 1.3", "MP": "13.7 ± 1.3", "p Value": ".01", "Measure": "Vascular density (%)"},
    {"Control": "Not provided", "MP": "Not provided", "p Value": "Not provided", "Measure": "Total vessel surface area (μm²)"},
]

st.set_page_config(
    page_title="ScaffoldScope | Vascular Analysis",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --bg:#06111f; --panel:#0b1929; --line:#203950; --cyan:#24d9e8; --muted:#9bb2ca; --text:#edf6ff; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background:radial-gradient(ellipse at 80% 0%,#102641 0%,#071321 44%,#050d18 100%); color:var(--text); }
[data-testid="stHeader"] { background:rgba(5,13,24,.7); }
.block-container { padding-top:1.4rem; padding-bottom:2rem; max-width:1600px; }
h1,h2,h3 { font-family:'Manrope',sans-serif !important; color:var(--text) !important; letter-spacing:-.03em; }
h1 { font-size:2.25rem !important; font-weight:800 !important; }
p, label, .stMarkdown { color:#dce9f8; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#0a1727 0%,#07111e 100%); border-right:1px solid #193047; }
[data-testid="stSidebar"] > div { padding-top:1.3rem; }
.brand { font-family:'Manrope',sans-serif; font-weight:800; font-size:1.55rem; letter-spacing:-.06em; color:#f2f8ff; }
.brand span { color:var(--cyan); }
.brand-sub { font-size:.68rem; color:#9db5d0; letter-spacing:.18em; text-transform:uppercase; margin-bottom:1.5rem; }
.eyebrow { color:var(--cyan); font-size:.72rem; font-weight:700; letter-spacing:.15em; text-transform:uppercase; }
.hero-sub { color:#9eb8d5; font-size:1.02rem; margin-top:-.6rem; }
.panel { background:linear-gradient(145deg,rgba(14,31,50,.96),rgba(8,20,34,.98)); border:1px solid #203950; border-radius:16px; padding:1rem 1.1rem; box-shadow:0 12px 35px rgba(0,0,0,.12); }
.panel-title { color:#eef7ff; font-size:1rem; font-weight:700; margin-bottom:.8rem; }
.metric-card { border:1px solid #21445a; background:linear-gradient(145deg,#0e2638,#0a1929); border-radius:14px; padding:1rem; min-height:115px; }
.metric-label { font-size:.79rem; color:#a9bfd6; margin-bottom:.55rem; }
.metric-value { font-family:'Manrope',sans-serif; font-size:1.55rem; font-weight:800; color:#f3f9ff; letter-spacing:-.04em; }
.metric-foot { font-size:.72rem; color:#89a5c1; margin-top:.3rem; }
.cyan { border-color:#087e94; } .green { border-color:#087d68; } .coral { border-color:#994353; } .purple { border-color:#6742a0; }
.status { display:inline-block; padding:.32rem .62rem; border-radius:999px; color:#58e6c0; background:#10372f; border:1px solid #1d6758; font-size:.75rem; font-weight:700; }
[data-testid="stFileUploader"] { background:rgba(7,29,45,.72); border:1px dashed #1aa5bf; border-radius:13px; padding:.5rem; }
.stButton > button, .stDownloadButton > button { border-radius:10px; border:1px solid #18b7cc; background:linear-gradient(135deg,#1dd2df,#1aaac6); color:#031521; font-weight:700; min-height:2.5rem; }
[data-testid="stMetric"] { background:#0b1c2d; border:1px solid #1c3a50; padding:.8rem; border-radius:12px; }
[data-testid="stMetricLabel"] { color:#a8c1d9 !important; }
[data-testid="stMetricValue"] { color:#f3f8ff !important; }
[data-testid="stDataFrame"] { border:1px solid #203950; border-radius:12px; overflow:hidden; }
hr { border-color:#20374b; }
.small-muted { color:#8ea8c2; font-size:.78rem; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div style="font-size:2rem;margin-bottom:.2rem">✳️</div><div class="brand">Scaffold<span>Scope</span></div><div class="brand-sub">Vascular image analysis</div>', unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("WORKSPACE", ["Overview", "Analyze Image", "Reference Comparison", "Model Status"])
    st.markdown("---")
    st.markdown('<span class="status">● Local workspace</span>', unsafe_allow_html=True)
    st.caption("Research-use estimates · validate before scientific interpretation.")
    st.markdown("---")
    st.markdown("**Image calibration**")
    pixels_per_um = st.number_input(
        "Pixels per micrometer",
        min_value=0.01,
        value=PIXELS_PER_UM_DEFAULT,
        step=0.1,
        help="You provided 1.1 pixels per micrometer. Change this only if your image calibration differs.",
    )
    um_per_pixel = 1.0 / pixels_per_um
    st.caption(f"Scale: {um_per_pixel:.4f} μm/pixel")

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

if page == "Reference Comparison":
    st.markdown('<div class="panel-title">Control vs MP · supplied reference values</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel">These values are the group-level values you supplied. They are displayed as a reference table and are not calculated from the uploaded image.</div>', unsafe_allow_html=True)
    ref_df = pd.DataFrame(REFERENCE_ROWS)[["Measure", "Control", "MP", "p Value"]]
    st.dataframe(ref_df, use_container_width=True, hide_index=True)
    st.download_button(
        "↓ Download reference table (CSV)",
        data=ref_df.to_csv(index=False).encode("utf-8"),
        file_name="scaffoldscope_control_vs_mp_reference.csv",
        mime="text/csv",
    )
    st.info("The supplied p value for average vessel diameter is shown as “.00” exactly as provided. Confirm the source convention before publication; a p value is not literally zero. Total vessel surface area values and its p value were not supplied.")
    st.stop()

if page == "Overview":
    st.markdown('<div class="panel"><div class="panel-title">Welcome to ScaffoldScope</div>Upload a microscopy image to review a vessel-mask prediction and morphology measurements. Model-based segmentation requires a trained checkpoint.<div class="small-muted" style="margin-top:.5rem">Calibration is set to 1.1 pixels/μm from your message.</div></div>', unsafe_allow_html=True)
elif page == "Analyze Image":
    st.markdown('<div class="eyebrow">NEW ANALYSIS</div>', unsafe_allow_html=True)

st.markdown('<div class="panel-title">01 / Select microscopy image</div>', unsafe_allow_html=True)
file = st.file_uploader("Drop a microscopy image here or browse files", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"])

if not file:
    if model is None:
        st.warning("No trained model checkpoint found. The interface is ready, but model-based predictions require training.")
        st.code("./train_mac.sh 20", language="bash")
    c1, c2, c3 = st.columns(3)
    for col, n, title, desc in [
        (c1, "STEP 01", "Upload", "Choose a microscopy image."),
        (c2, "STEP 02", "Segment", "Run the trained U-Net to estimate vessel pixels."),
        (c3, "STEP 03", "Measure & export", "Review measurements and export results."),
    ]:
        with col:
            st.markdown(f'<div class="panel"><div class="eyebrow">{n}</div><div class="panel-title">{title}</div><div class="small-muted">{desc}</div></div>', unsafe_allow_html=True)
    st.markdown("---")
    st.subheader("Reference group comparison")
    st.caption("Supplied Control and MP values. These are reference data, not measurements from an uploaded image.")
    st.dataframe(pd.DataFrame(REFERENCE_ROWS)[["Measure", "Control", "MP", "p Value"]], use_container_width=True, hide_index=True)
    st.stop()

original = Image.open(file).convert("L")
arr = np.asarray(original)
if model is None:
    st.image(original, caption="Input microscopy image", use_container_width=True)
    st.warning("A trained checkpoint is required for vessel segmentation. No prediction or image-specific measurements are fabricated.")
    st.stop()

threshold = st.sidebar.slider("Segmentation threshold", 0.05, 0.95, 0.50, 0.05)
min_obj = st.sidebar.slider("Remove objects smaller than (pixels)", 1, 100, 8)
device, size = info
resized = original.resize((size, size), Image.Resampling.BILINEAR)
x = np.asarray(resized, dtype=np.float32) / 255.0
with torch.no_grad():
    prob = torch.sigmoid(model(torch.from_numpy(x[None, None]).to(device)))[0, 0].detach().cpu().numpy()
prob_img = Image.fromarray(np.uint8(np.clip(prob, 0, 1) * 255)).resize(original.size, Image.Resampling.BILINEAR)
prob_full = np.asarray(prob_img, dtype=np.float32) / 255.0
stats = analyze(prob_full >= threshold, um_per_pixel, min_obj)

overlay = np.stack([arr, arr, arr], axis=-1).astype(np.float32)
mask = stats["mask"].astype(bool)
overlay[mask] = 0.50 * overlay[mask] + 0.50 * np.array([20, 220, 235], dtype=np.float32)

st.markdown(f'<div class="panel-title">02 / Image review <span class="small-muted">· {file.name}</span></div>', unsafe_allow_html=True)
image_col, metric_col = st.columns([1.2, 1])
with image_col:
    st.markdown('<div class="panel-title">Image preview</div>', unsafe_allow_html=True)
    t1, t2, t3 = st.tabs(["Original", "Segmentation overlay", "Binary mask"])
    with t1:
        st.image(original, use_container_width=True)
    with t2:
        st.image(np.uint8(np.clip(overlay, 0, 255)), use_container_width=True)
    with t3:
        st.image(np.uint8(mask * 255), use_container_width=True)

with metric_col:
    st.markdown('<div class="panel-title">Image-specific analysis summary</div>', unsafe_allow_html=True)
    m1, m2 = st.columns(2)
    with m1:
        st.markdown(f'<div class="metric-card green"><div class="metric-label">Vascular area</div><div class="metric-value">{stats["area_pixels"]:,}</div><div class="metric-foot">foreground pixels</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card cyan"><div class="metric-label">Connected components</div><div class="metric-value">{stats["components"]:,}</div><div class="metric-foot">detected regions</div></div>', unsafe_allow_html=True)
    st.write("")
    m3, m4 = st.columns(2)
    with m3:
        st.markdown(f'<div class="metric-card coral"><div class="metric-label">Skeleton endpoints</div><div class="metric-value">{stats["endpoints"]:,}</div><div class="metric-foot">estimated endpoints</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="metric-card purple"><div class="metric-label">Approx. skeleton length</div><div class="metric-value">{stats["approx_length_pixels"] / pixels_per_um:,.1f}</div><div class="metric-foot">approximate μm</div></div>', unsafe_allow_html=True)

st.write("")
chart_col, export_col = st.columns([1.2, 1])
with chart_col:
    st.markdown('<div class="panel-title">Vessel morphology</div>', unsafe_allow_html=True)
    st.caption("Vessel-pixel fraction by image row. This is a visualization aid, not a diameter distribution.")
    st.line_chart({"Vessel pixel fraction by row": mask.mean(axis=1)}, height=220)
with export_col:
    st.markdown('<div class="panel-title">Export results</div>', unsafe_allow_html=True)
    st.caption("Download image-specific measurements and the binary mask.")
    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow(["filename", "threshold", "min_object_pixels", "pixels_per_micrometer", "micrometers_per_pixel", "area_pixels", "area_um2_estimate", "components", "endpoints", "approx_length_pixels", "approx_length_um_estimate"])
    writer.writerow([file.name, threshold, min_obj, pixels_per_um, um_per_pixel, stats.get("area_pixels"), stats.get("area_pixels") * (um_per_pixel ** 2), stats.get("components"), stats.get("endpoints"), stats.get("approx_length_pixels"), stats.get("approx_length_pixels") * um_per_pixel])
    st.download_button("↓ Download measurements (CSV)", csv_buffer.getvalue(), file_name=f"{Path(file.name).stem}_measurements.csv", mime="text/csv", use_container_width=True)
    mask_buffer = io.BytesIO()
    Image.fromarray(np.uint8(mask * 255)).save(mask_buffer, format="PNG")
    st.download_button("↓ Download segmentation mask (PNG)", mask_buffer.getvalue(), file_name=f"{Path(file.name).stem}_mask.png", mime="image/png", use_container_width=True)
    st.caption(f"Calibration applied: {pixels_per_um:g} pixels/μm ({um_per_pixel:.4f} μm/pixel).")

st.markdown("---")
st.subheader("Control vs MP reference comparison")
st.caption("Group-level values supplied by you; not calculated from this image.")
st.dataframe(pd.DataFrame(REFERENCE_ROWS)[["Measure", "Control", "MP", "p Value"]], use_container_width=True, hide_index=True)
st.markdown('<div class="small-muted">Research-use estimates only. The current model does not directly calculate loop count, mean vessel diameter, vascular density percentage, or total vessel surface area. Validate segmentation and measurements against calibrated reference data before scientific use.</div>', unsafe_allow_html=True)
