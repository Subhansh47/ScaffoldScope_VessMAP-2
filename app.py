from pathlib import Path
import io
import csv

import numpy as np
import streamlit as st
import torch
from PIL import Image
from scipy import ndimage as ndi
from skimage.morphology import skeletonize, remove_small_objects
from skimage.measure import label
from scaffoldscope.model import UNet

ROOT = Path(__file__).resolve().parent
CKPT = ROOT / "models" / "scaffoldscope_unet.pt"

st.set_page_config(page_title="ScaffoldScope | 2D Vascular Analysis", page_icon="🧬", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;600;700;800&display=swap');
.stApp{background:radial-gradient(ellipse at 80% 0%,#102641 0%,#071321 44%,#050d18 100%);color:#edf6ff}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
.block-container{padding-top:1.4rem;max-width:1550px}
h1,h2,h3{font-family:'Manrope',sans-serif!important;color:#edf6ff!important}
[data-testid="stSidebar"]{background:#081522;border-right:1px solid #203950}
.eyebrow{color:#24d9e8;font-size:.72rem;font-weight:700;letter-spacing:.15em}
.sub{color:#9eb8d5}
.card{background:linear-gradient(145deg,#0e1f32,#081422);border:1px solid #203950;border-radius:15px;padding:1rem;min-height:112px}
.lab{font-size:.8rem;color:#a9bfd6}.val{font-family:'Manrope',sans-serif;font-size:1.5rem;font-weight:800;color:#f3f9ff}.foot{font-size:.72rem;color:#89a5c1}
.stButton>button,.stDownloadButton>button{border-radius:10px;border:1px solid #18b7cc;background:linear-gradient(135deg,#1dd2df,#1aaac6);color:#031521;font-weight:700}
[data-testid="stFileUploader"]{background:#071d2d;border:1px dashed #1aa5bf;border-radius:13px}
hr{border-color:#20374b}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div style="font-size:2rem">✳️</div><h2 style="margin:0">Scaffold<span style="color:#24d9e8">Scope</span></h2><div class="sub">2D vascular image analysis</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### Image calibration")
    px_per_um = st.number_input("Pixels per micrometer", min_value=0.01, value=1.1, step=0.1,
        help="Using your supplied calibration: 1.1 pixels = 1 micrometer.")
    um_per_px = 1.0 / px_per_um
    st.caption(f"{um_per_px:.4f} μm per pixel")
    threshold = st.slider("Segmentation threshold", 0.05, 0.95, 0.50, 0.05)
    min_obj = st.slider("Remove objects smaller than (pixels)", 1, 100, 8)
    st.markdown("---")
    st.caption("Research-use estimates. Confirm image calibration and validate measurements before reporting.")

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

def measure_network(binary, um_per_px, min_pixels):
    mask = remove_small_objects(binary.astype(bool), min_size=int(min_pixels), connectivity=2)
    skel = skeletonize(mask)
    # Eight-neighbor degree map; cluster adjacent branch pixels to reduce junction over-counting.
    kernel = np.ones((3, 3), dtype=np.uint8)
    neighbor_count = ndi.convolve(skel.astype(np.uint8), kernel, mode="constant", cval=0) - skel.astype(np.uint8)
    branch_pixels = skel & (neighbor_count >= 3)
    branch_clusters, branch_count = ndi.label(branch_pixels, structure=np.ones((3,3), dtype=np.uint8))
    endpoints = int(np.count_nonzero(skel & (neighbor_count == 1)))
    components = int(label(mask, connectivity=2).max()) if mask.any() else 0

    # 8-connected skeleton length: horizontal/vertical steps weigh 1, diagonal steps sqrt(2).
    sk = skel.astype(np.uint8)
    horizontal = int(np.count_nonzero(sk[:, :-1] & sk[:, 1:]))
    vertical = int(np.count_nonzero(sk[:-1, :] & sk[1:, :]))
    diagonal_a = int(np.count_nonzero(sk[:-1, :-1] & sk[1:, 1:]))
    diagonal_b = int(np.count_nonzero(sk[:-1, 1:] & sk[1:, :-1]))
    length_px_rough = max(0.0, (horizontal + vertical + np.sqrt(2) * (diagonal_a + diagonal_b)) / 2.0)
    # Diameter estimated from the Euclidean distance transform at skeleton pixels.
    dist = ndi.distance_transform_edt(mask)
    diameters_px = 2.0 * dist[skel]
    mean_diameter_um = float(np.mean(diameters_px) * um_per_px) if diameters_px.size else 0.0
    median_diameter_um = float(np.median(diameters_px) * um_per_px) if diameters_px.size else 0.0

    # In a 2D binary image, holes give the independent loop count (Euler topology).
    filled = ndi.binary_fill_holes(mask)
    holes = int(max(0, label(filled, connectivity=2).max() - components)) if mask.any() else 0
    # More direct hole count: connected components of the complement not touching the border.
    complement_labels, n_complement = ndi.label(~mask, structure=np.ones((3,3), dtype=np.uint8))
    border_ids = np.unique(np.concatenate([complement_labels[0,:], complement_labels[-1,:], complement_labels[:,0], complement_labels[:,-1]]))
    hole_ids = set(range(1, n_complement + 1)) - set(border_ids.tolist())
    loops = len(hole_ids)

    area_px = int(mask.sum())
    total_px = int(mask.size)
    density_pct = (100.0 * area_px / total_px) if total_px else 0.0
    vessel_area_um2 = area_px * (um_per_px ** 2)
    length_um = length_px_rough * um_per_px
    surface_area_um2 = None  # Cannot recover 3D vessel surface area from a single 2D image.
    return {
        "mask": mask, "skeleton": skel, "branch_count": int(branch_count),
        "loop_count": int(loops), "endpoints": endpoints, "components": components,
        "area_pixels": area_px, "image_pixels": total_px, "density_pct": density_pct,
        "length_px": length_px_rough, "length_um": length_um,
        "mean_diameter_um": mean_diameter_um, "median_diameter_um": median_diameter_um,
        "vessel_area_um2": vessel_area_um2, "surface_area_um2": surface_area_um2
    }

def make_annotated_overlay(gray, results, um_per_px, show_endpoints=True, show_branches=True, show_lengths=True, show_diameters=True):
    """Draw every skeleton component and label each component's estimated length."""
    from PIL import ImageDraw, ImageFont
    mask, skel = results["mask"], results["skeleton"]
    # Preserve grayscale detail and tint only predicted vessel pixels.
    raw = np.asarray(gray.convert("RGB"), dtype=np.float32)
    base = np.clip(raw * 0.72 + np.array([4, 10, 18]), 0, 255).astype(np.uint8)
    base[mask] = np.clip(0.36 * raw[mask] + 0.64 * np.array([20, 210, 225]), 0, 255).astype(np.uint8)
    image = Image.fromarray(base)
    d = ImageDraw.Draw(image)
    scale = max(1.0, gray.width / 900.0)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", max(12, int(16*scale)))
        small = ImageFont.truetype("DejaVuSans.ttf", max(10, int(12*scale)))
    except OSError:
        font = small = ImageFont.load_default()
    ys, xs = np.nonzero(skel)
    for x, y in zip(xs, ys):
        d.point((int(x), int(y)), fill=(35, 255, 135))

    kernel = np.ones((3, 3), dtype=np.uint8)
    degree = ndi.convolve(skel.astype(np.uint8), kernel, mode="constant") - skel.astype(np.uint8)
    endpoints = np.argwhere(skel & (degree == 1))
    bp = skel & (degree >= 3)
    labels, n = ndi.label(bp, structure=kernel)
    centers = []
    for i in range(1, n + 1):
        pts = np.argwhere(labels == i)
        if len(pts):
            y, x = pts.mean(axis=0)
            centers.append((int(x), int(y)))
    r = max(3, int(gray.width / 220))
    if show_endpoints:
        for y, x in endpoints:
            d.ellipse((int(x-r), int(y-r), int(x+r), int(y+r)), fill=(0, 160, 255), outline="white", width=max(1, int(scale)))
    if show_branches:
        for x, y in centers:
            d.ellipse((x-r-1, y-r-1, x+r+1, y+r+1), fill=(255, 225, 0), outline=(10, 20, 30), width=max(1, int(scale)))

    # Label each disconnected skeleton structure, rather than highlighting only one path.
    # Length is the sum of 8-neighbour centerline steps, estimated per structure.
    comp_labels, ncomp = ndi.label(skel, structure=kernel)
    component_rows = []
    if show_lengths:
        for ci in range(1, ncomp + 1):
            comp = comp_labels == ci
            coords = np.argwhere(comp)
            if len(coords) < 2:
                continue
            yy, xx = np.nonzero(comp)
            local = comp.astype(np.uint8)
            h = int(np.count_nonzero(local[:, :-1] & local[:, 1:]))
            v = int(np.count_nonzero(local[:-1, :] & local[1:, :]))
            da = int(np.count_nonzero(local[:-1, :-1] & local[1:, 1:]))
            db = int(np.count_nonzero(local[:-1, 1:] & local[1:, :-1]))
            # Divide adjacency sum by two because every pixel pair is counted from both ends.
            length_px = (h + v + np.sqrt(2) * (da + db)) / 2.0
            length_um = length_px * um_per_px
            cy, cx = np.mean(yy), np.mean(xx)
            component_rows.append({"component": ci, "length_um": float(length_um), "pixels": int(len(coords)), "center_x": int(cx), "center_y": int(cy)})
            label_text = f"V{ci}: {length_um:.1f} um"
            tx = int(np.clip(cx + 7, 2, gray.width - 160))
            ty = int(np.clip(cy + 7, 2, gray.height - 28))
            box = d.textbbox((tx, ty), label_text, font=small, stroke_width=0)
            d.rounded_rectangle((box[0]-4, box[1]-3, box[2]+4, box[3]+3), radius=4, fill=(2, 18, 30), outline=(0, 220, 240), width=1)
            d.text((tx, ty), label_text, font=small, fill=(105, 255, 250))

    dist = ndi.distance_transform_edt(mask)
    if show_diameters and skel.any():
        cy, cx = np.unravel_index(np.argmax(dist * skel), dist.shape)
        rad = float(dist[cy, cx])
        if rad > 0:
            d.line((int(cx-rad), int(cy), int(cx+rad), int(cy)), fill=(255, 155, 35), width=max(2, int(2*scale)))
            text = f"Diameter ~ {2*rad*um_per_px:.1f} um"
            tx = int(np.clip(cx + rad + 6, 2, gray.width-190)); ty = int(np.clip(cy-rad*3, 2, gray.height-25))
            d.text((tx, ty), text, font=small, fill=(255, 190, 80), stroke_width=2, stroke_fill=(0, 0, 0))

    # Compact legend with readable type, sized to the actual image.
    legend = [((20, 210, 225), "Cyan overlay: predicted vessel"), ((35, 255, 135), "Green: skeleton centerline"), ((0, 160, 255), "Blue: endpoint"), ((255, 225, 0), "Yellow: branch point"), ((255, 155, 35), "Orange: diameter"), ((105, 255, 250), "V# label: length of each disconnected skeleton")]
    line_h = max(19, int(22*scale)); pad = 9
    bw = min(gray.width-16, max(290, int(gray.width*0.44)))
    bh = pad*2 + line_h*len(legend)
    d.rounded_rectangle((8, 8, bw, 8+bh), radius=8, fill=(3, 15, 28), outline=(25, 180, 205), width=2)
    for i, (color, label_text) in enumerate(legend):
        y = 8 + pad + i*line_h
        d.line((18, y+7, 36, y+7), fill=color, width=max(2, int(3*scale)))
        d.text((43, y), label_text, font=small, fill=(240, 248, 255))
    return image, component_rows

model, model_info = load_model()
st.markdown('<div class="eyebrow">MICROSCOPY INTELLIGENCE WORKSPACE</div>', unsafe_allow_html=True)
st.title("Vascular Analysis")
st.markdown('<div class="sub">Upload a 2D microscopy image to segment vessels and estimate network morphology.</div>', unsafe_allow_html=True)
if model is None:
    st.warning("No trained checkpoint found at models/scaffoldscope_unet.pt. The app can be opened, but image predictions require a trained model.")
    st.code("./train_mac.sh 20", language="bash")

uploaded = st.file_uploader("Upload microscopy image", type=["png","jpg","jpeg","tif","tiff","bmp"])
if not uploaded:
    a,b,c = st.columns(3)
    for col, step, title, detail in [
        (a,"01","Upload image","Choose a 2D microscopy image."),
        (b,"02","Segment vessels","The trained U-Net predicts vessel pixels."),
        (c,"03","Measure network","Estimate branch points, loops, length, diameter, and density.")
    ]:
        with col:
            st.markdown(f'<div class="card"><div class="eyebrow">{step}</div><h3>{title}</h3><div class="sub">{detail}</div></div>', unsafe_allow_html=True)
    st.stop()

original = Image.open(uploaded).convert("L")
arr = np.asarray(original)
st.caption(f"Image: {uploaded.name} · {original.width} × {original.height} pixels · calibration {px_per_um:g} px/μm")
if model is None:
    st.image(original, caption="Uploaded image (prediction unavailable until a trained checkpoint exists)", use_container_width=True)
    st.stop()

device, model_size = model_info
resized = original.resize((model_size, model_size), Image.Resampling.BILINEAR)
x = np.asarray(resized, dtype=np.float32) / 255.0
with torch.no_grad():
    probability = torch.sigmoid(model(torch.from_numpy(x[None, None]).to(device)))[0, 0].detach().cpu().numpy()
prob_img = Image.fromarray(np.uint8(np.clip(probability, 0, 1) * 255)).resize(original.size, Image.Resampling.BILINEAR)
prob_full = np.asarray(prob_img, dtype=np.float32) / 255.0
results = measure_network(prob_full >= threshold, um_per_px, min_obj)
mask = results["mask"]
with st.expander("Annotation controls", expanded=True):
    ac1, ac2, ac3, ac4 = st.columns(4)
    show_endpoints = ac1.checkbox("Show endpoints", True)
    show_branches = ac2.checkbox("Show branch points", True)
    show_lengths = ac3.checkbox("Show segment length", True)
    show_diameters = ac4.checkbox("Show diameter", True)
annotated_overlay, component_rows = make_annotated_overlay(original, results, um_per_px, show_endpoints, show_branches, show_lengths, show_diameters)

left, right = st.columns([1.15, 1])
with left:
    st.subheader("Image review")
    tab1, tab2, tab3 = st.tabs(["Original", "Segmentation overlay", "Binary mask"])
    with tab1: st.image(original, use_container_width=True)
    with tab2:
        st.image(annotated_overlay, caption="Annotated segmentation generated from this image's predicted vessel mask.", use_container_width=True)
        st.caption("Every disconnected skeleton structure is labeled V1, V2, etc. with its estimated centerline length. Branch-connected networks may form one structure, so individual branch-to-branch edges are not yet separately reported. Diameter is a local distance-transform estimate.")
    with tab3: st.image(np.uint8(mask * 255), use_container_width=True)
with right:
    st.subheader("Predicted morphology")
    metrics = [
        ("Branch count", f'{results["branch_count"]}', "clustered skeleton junction estimate"),
        ("Loop count", f'{results["loop_count"]}', "2D enclosed-loop estimate"),
        ("Total tube length", f'{results["length_um"]:,.1f} μm', "approximate skeleton length"),
        ("Mean vessel diameter", f'{results["mean_diameter_um"]:.2f} μm', "distance-transform estimate"),
        ("Vascular density", f'{results["density_pct"]:.2f}%', "vessel pixels / full image pixels"),
        ("Vessel area (2D)", f'{results["vessel_area_um2"]:,.1f} μm²', "area in this 2D image, not 3D surface area"),
    ]
    for i in range(0, len(metrics), 2):
        m1,m2 = st.columns(2)
        for col, item in zip((m1,m2), metrics[i:i+2]):
            with col:
                st.markdown(f'<div class="card"><div class="lab">{item[0]}</div><div class="val">{item[1]}</div><div class="foot">{item[2]}</div></div>', unsafe_allow_html=True)
        st.write("")

st.markdown("---")
st.subheader("Measurement details")
detail_df = [
    {"Measurement":"Image dimensions","Result":f"{original.width} × {original.height} px","Method":"Uploaded image size"},
    {"Measurement":"Calibration","Result":f"{px_per_um:g} px/μm; {um_per_px:.4f} μm/px","Method":"User-supplied calibration"},
    {"Measurement":"Branch count","Result":results["branch_count"],"Method":"Clusters of skeleton pixels with ≥3 neighbors"},
    {"Measurement":"Loop count","Result":results["loop_count"],"Method":"Enclosed background regions in 2D mask"},
    {"Measurement":"Total tube length","Result":f'{results["length_um"]:.2f} μm',"Method":"Approximate 8-neighbor skeleton length"},
    {"Measurement":"Average vessel diameter","Result":f'{results["mean_diameter_um"]:.2f} μm',"Method":"Mean 2×distance-transform radius at skeleton pixels"},
    {"Measurement":"Vascular density","Result":f'{results["density_pct"]:.2f}%',"Method":"Mask area divided by full image area"},
    {"Measurement":"Total vessel area (2D)","Result":f'{results["vessel_area_um2"]:.2f} μm²',"Method":"Vessel foreground pixel count × pixel area"},
    {"Measurement":"Total vessel surface area (3D)","Result":"Not measurable from a single 2D image","Method":"Requires 3D volume or explicit geometric assumptions"},
]
import pandas as pd
st.dataframe(pd.DataFrame(detail_df), use_container_width=True, hide_index=True)

if component_rows:
    st.subheader("Vessel structure lengths shown on the overlay")
    st.caption("Lengths are estimated from the skeleton centerline. Adjacent branches connected at junctions may be grouped as one structure.")
    st.dataframe(pd.DataFrame([{ "Overlay label": f"V{r['component']}", "Estimated length (μm)": round(r['length_um'], 2), "Skeleton pixels": r['pixels']} for r in component_rows]), use_container_width=True, hide_index=True)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Skeleton preview")
    st.image(np.uint8(results["skeleton"] * 255), caption="Thinned vessel skeleton used for length and keypoint estimates.", use_container_width=True)
    st.markdown("**How to read the annotations**")
    st.markdown("- 🔵 Blue circles mark detected skeleton endpoints.\n- 🟡 Yellow circles mark candidate branch junctions.\n- 🟢 Green pixels trace the skeleton centerline.\n- 🩵 Each V# label reports the estimated length of a disconnected skeleton structure.\n- 🟠 Orange line shows one local diameter estimate.\n- A connected branched network is currently measured as one structure, not as separate branch-to-branch segments.")
with col2:
    st.subheader("Export")
    csv_rows = [
        ["filename", uploaded.name],
        ["image_width_px", original.width],
        ["image_height_px", original.height],
        ["pixels_per_micrometer", px_per_um],
        ["micrometers_per_pixel", um_per_px],
        ["branch_count_estimate", results["branch_count"]],
        ["loop_count_estimate", results["loop_count"]],
        ["total_tube_length_um_estimate", results["length_um"]],
        ["mean_vessel_diameter_um_estimate", results["mean_diameter_um"]],
        ["median_vessel_diameter_um_estimate", results["median_diameter_um"]],
        ["vascular_density_percent", results["density_pct"]],
        ["vessel_area_2d_um2", results["vessel_area_um2"]],
        ["vessel_surface_area_3d_um2", "not measurable from a single 2D image"],
    ]
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["metric","value"])
    writer.writerows(csv_rows)
    st.download_button("Download measurements (CSV)", buf.getvalue(), file_name=f"{Path(uploaded.name).stem}_scaffoldscope_metrics.csv", mime="text/csv", use_container_width=True)
    png = io.BytesIO()
    Image.fromarray(np.uint8(mask * 255)).save(png, format="PNG")
    st.download_button("Download vessel mask (PNG)", png.getvalue(), file_name=f"{Path(uploaded.name).stem}_vessel_mask.png", mime="image/png", use_container_width=True)

st.info("These are algorithmic estimates from the model's segmentation, not trained predictions of the reference-table group means. Branch/loop counting and 2D geometry require validation against manually annotated images. A single 2D image cannot establish true 3D vessel surface area.")
