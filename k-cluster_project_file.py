import streamlit as st
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import os
import cv2
import random

# Streamlit page config
st.set_page_config(page_title="Image Compression - KMeans", layout="wide")
st.title("🖼 Image Compression with K-Means Clustering")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

# Compression quality selector
compression_level = st.radio(
    "Choose Compression Level",
    ("Small (High Compression)", "Medium", "Large (Best Quality)")
)

# Correct K mapping: Small = high K, Large = low K
k_map = {
    "Small (High Compression)": 8,    # Fewer colors → smaller size
    "Medium": 16,                     # Balanced
    "Large (Best Quality)": 32        # More colors → better quality
}
k = k_map[compression_level]

# Resize factor to reduce dimensions & file size
resize_factor_map = {
    "Small (High Compression)": 0.5,   # Reduce size by 50%
    "Medium": 0.75,                    # Reduce size by 25%
    "Large (Best Quality)": 1.0        # Keep original size
}
resize_factor = resize_factor_map[compression_level]

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file)

    # Resize if too large for faster processing
    max_size = 800
    scale = max(image.size) / max_size
    if scale > 1:
        new_size = (int(image.size[0] / scale), int(image.size[1] / scale))
        image = image.resize(new_size, Image.Resampling.LANCZOS)

    # Convert to numpy array
    img_np = np.array(image)

    # Reshape image for KMeans
    pixels = img_np.reshape(-1, 3)

    # Apply KMeans
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels)
    new_colors = kmeans.cluster_centers_.astype("uint8")
    compressed_img = new_colors[labels].reshape(img_np.shape)

    # Downscale compressed image to reduce file size further
    if resize_factor < 1.0:
        new_size = (int(compressed_img.shape[1] * resize_factor),
                    int(compressed_img.shape[0] * resize_factor))
        compressed_img = cv2.resize(compressed_img, new_size, interpolation=cv2.INTER_AREA)

    # Show original and compressed images in medium size
    col1, col2 = st.columns(2)
    col1.image(img_np, caption="Original Image", width=350)
    col2.image(compressed_img, caption=f"Compressed Image (K={k})", width=350)

    # Save compressed image temporarily
    compressed_image_pil = Image.fromarray(compressed_img)
    compressed_image_pil.save("compressed_image.jpg", format="JPEG", quality=85, optimize=True)

    # Calculate file sizes
    orig_size_kb = uploaded_file.size / 1024
    comp_size_kb = os.path.getsize("compressed_image.jpg") / 1024
    reduction_percent = 100 - (comp_size_kb / orig_size_kb * 100)

    st.markdown(
        f"**Original Size:** {orig_size_kb:.2f} KB  |  "
        f"**Compressed Size:** {comp_size_kb:.2f} KB  |  "
        f"**Reduction:** {reduction_percent:.1f}%"
    )

    # Generate cluster visualization
    overlay_colors = np.array([
        [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
        for _ in range(k)
    ])
    cluster_overlay = overlay_colors[labels].reshape(img_np.shape).astype(np.uint8)
    compressed_img_vis = new_colors[labels].reshape(img_np.shape).astype(np.uint8)
    alpha = 0.4  # Transparency for overlay
    highlighted_clusters = cv2.addWeighted(compressed_img_vis, 1 - alpha, cluster_overlay, alpha, 0)

    

    # Download option
    with open("compressed_image.jpg", "rb") as file:
        st.download_button("📥 Download Compressed Image", file, file_name="compressed_image.jpg")
