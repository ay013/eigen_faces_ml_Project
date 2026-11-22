import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA

st.set_page_config(page_title="Eigenfaces PCA Demo", layout="wide")

# ---------------------------------------------------------
# Load Dataset & Fit Full PCA Once (Cached for Instant Slider)
# ---------------------------------------------------------
@st.cache_data
def load_and_fit_pca():
    faces = fetch_olivetti_faces(shuffle=True, random_state=42)
    X = faces.data  # (400, 4096), values in [0, 1]
    pca = PCA(svd_solver="randomized", random_state=42)
    pca.fit(X)
    return X, pca

X, pca = load_and_fit_pca()
IMAGE_SHAPE = (64, 64)
MAX_K = 350

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
st.sidebar.title("PCA Controls")
k = st.sidebar.slider(
    "Number of Eigenfaces (k)",
    min_value=1,
    max_value=MAX_K,
    value=50,
    step=1,
)

input_mode = st.sidebar.radio(
    "Image Source",
    ["Dataset Face (Olivetti)", "Upload Custom Image"],
)

if input_mode == "Dataset Face (Olivetti)":
    face_idx = st.sidebar.slider("Select Face Index", 0, len(X) - 1, 0)
    original_flat = X[face_idx]
else:
    uploaded = st.sidebar.file_uploader(
        "Upload a face (or non-face) photo", type=["jpg", "jpeg", "png"]
    )
    if uploaded is not None:
        img = Image.open(uploaded).convert("L").resize(IMAGE_SHAPE)
        original_flat = np.asarray(img, dtype=np.float32).flatten() / 255.0
    else:
        st.sidebar.info("Upload an image to test, or showing default face #0.")
        original_flat = X[0]

# ---------------------------------------------------------
# Instant Projection & Reconstruction using Top-k Components
# ---------------------------------------------------------
mean_face = pca.mean_
components_k = pca.components_[:k]

centered = original_flat - mean_face
weights = centered @ components_k.T               # Compressed vector (k,)
reconstructed_flat = mean_face + weights @ components_k
reconstructed_clipped = np.clip(reconstructed_flat, 0.0, 1.0)

# Metrics
mse = float(np.mean((original_flat - reconstructed_flat) ** 2))
var_retained = float(np.sum(pca.explained_variance_ratio_[:k]) * 100)
compression_ratio = 4096 / k

# ---------------------------------------------------------
# Main Header & Live Metric Cards
# ---------------------------------------------------------
st.title("Eigenfaces: Face Compression & Reconstruction (PCA)")
st.caption("Unsupervised Learning — Module IV | Dimensionality Reduction via Principal Component Analysis")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Components Used (k)", f"{k} / 4096")
m2.metric("Compression Ratio", f"{compression_ratio:.1f}x")
m3.metric("Variance Retained", f"{var_retained:.2f}%")
m4.metric("Reconstruction MSE", f"{mse:.5f}")

# ---------------------------------------------------------
# Interactive Tabs for Viva Presentation
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "1. Live Reconstruction",
    "2. Mean Face & Eigenfaces",
    "3. Explained Variance Curve",
])

with tab1:
    diff_map = np.abs(original_flat - reconstructed_clipped).reshape(IMAGE_SHAPE)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    axes[0].imshow(original_flat.reshape(IMAGE_SHAPE), cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Original (4,096 pixels)")
    axes[0].axis("off")

    axes[1].imshow(reconstructed_clipped.reshape(IMAGE_SHAPE), cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"Reconstructed (k = {k})")
    axes[1].axis("off")

    im = axes[2].imshow(diff_map, cmap="inferno", vmin=0, vmax=0.5)
    axes[2].set_title("Absolute Error Map")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    st.pyplot(fig)

    st.info(
        f"**Formula:** Reconstructed Face = Mean Face + Σ (wᵢ · Eigenfaceᵢ) for i = 1 to {k}. "
        "Notice how the error map fades to dark black as k increases."
    )

with tab2:
    col_mean, col_grid = st.columns([1, 3])

    with col_mean:
        st.subheader("Mean Face (μ)")
        fig_m, ax_m = plt.subplots(figsize=(3, 3))
        ax_m.imshow(mean_face.reshape(IMAGE_SHAPE), cmap="gray")
        ax_m.axis("off")
        st.pyplot(fig_m)

    with col_grid:
        st.subheader("Top 10 Eigenfaces (Principal Components)")
        fig_e, axes_e = plt.subplots(2, 5, figsize=(10, 4))
        for i, ax in enumerate(axes_e.flat):
            ax.imshow(pca.components_[i].reshape(IMAGE_SHAPE), cmap="gray")
            ax.set_title(f"PC {i + 1}", fontsize=9)
            ax.axis("off")
        plt.tight_layout()
        st.pyplot(fig_e)

with tab3:
    cum_var = np.cumsum(pca.explained_variance_ratio_) * 100
    k_95 = int(np.argmax(cum_var >= 95.0) + 1)

    fig_v, ax_v = plt.subplots(figsize=(8, 4))
    ax_v.plot(range(1, len(cum_var) + 1), cum_var, color="navy", linewidth=2, label="Cumulative Variance")
    ax_v.axhline(95, color="green", linestyle="--", label=f"95% Threshold (k = {k_95})")
    ax_v.scatter([k], [var_retained], color="red", s=80, zorder=5, label=f"Current k = {k} ({var_retained:.1f}%)")
    ax_v.set_xlim(1, MAX_K)
    ax_v.set_ylim(0, 102)
    ax_v.set_xlabel("Number of Principal Components (k)")
    ax_v.set_ylabel("Cumulative Explained Variance (%)")
    ax_v.set_title("Information Retained vs. Number of Eigenfaces")
    ax_v.legend()
    ax_v.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig_v)