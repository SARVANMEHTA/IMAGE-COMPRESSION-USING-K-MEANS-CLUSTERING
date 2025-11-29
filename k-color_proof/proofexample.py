import streamlit as st
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import matplotlib.pyplot as plt
import os

# Streamlit page settings
st.set_page_config(page_title="Image Compression - KMeans", layout="wide")
st.title("🖼 Image Compression with K-Means Clustering")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

# Compression quality selector
compression_level = st.radio(
    "Choose Compression Level",
    ("Small (High Compression)", "Medium", "Large (Best Quality)")
)

# Map compression levels to K values
k_map = {
    "Small (High Compression)": 8,    # Fewer colors → smaller size
    "Medium": 16,                     # Balanced
    "Large (Best Quality)": 32        # More colors → better quality
}
k = k_map[compression_level]

if uploaded_file is not None:
    # Read uploaded image
    image = Image.open(uploaded_file)

    # Resize if image is too large → faster processing
    max_size = 800
    scale = max(image.size) / max_size
    if scale > 1:
        new_size = (int(image.size[0] / scale), int(image.size[1] / scale))
        image = image.resize(new_size, Image.Resampling.LANCZOS)

    # Convert image to NumPy array
    img_np = np.array(image)

    # Reshape pixels for KMeans clustering
    pixels = img_np.reshape(-1, 3)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels)
    centroids = kmeans.cluster_centers_.astype("uint8")  # K centroid colors
    compressed_img = centroids[labels].reshape(img_np.shape)

    # Show Original & Compressed images side by side
    col1, col2 = st.columns(2)
    col1.image(img_np, caption="Original Image", width=350)
    col2.image(compressed_img, caption=f"Compressed Image (K={k})", width=350)

    # Save compressed image
    compressed_image_pil = Image.fromarray(compressed_img)
    compressed_image_pil.save("compressed_image.jpg", format="JPEG", quality=85, optimize=True)

    # Calculate file sizes
    orig_size_kb = uploaded_file.size / 1024
    comp_size_kb = os.path.getsize("compressed_image.jpg") / 1024
    reduction_percent = 100 - (comp_size_kb / orig_size_kb * 100)

    st.markdown(
        f"**Original Size:** {orig_size_kb:.2f} KB  |  **Compressed Size:** {comp_size_kb:.2f} KB  |  **Reduction:** {reduction_percent:.1f}%"
    )

    # ---------------------------
    # Proof 1 → Color Palette Visualization
    # ---------------------------
    st.subheader("🎨 K-Means Color Palette (Centroids)")
    fig1, ax1 = plt.subplots(figsize=(8, 2))
    for i, color in enumerate(centroids):
        ax1.add_patch(plt.Rectangle((i, 0), 1, 1, color=color / 255))
    ax1.set_xlim(0, len(centroids))
    ax1.set_ylim(0, 1)
    ax1.axis("off")
    st.pyplot(fig1)

    st.caption(f"Above are the **{k} centroid colors** selected by K-Means for compression.")

    # ---------------------------
    # Proof 2 → Color Histogram Visualization
    # ---------------------------
    st.subheader("📊 Color Histogram - Proof of K-Means Clustering")
    compressed_pixels = compressed_img.reshape(-1, 3)
    unique_colors, counts = np.unique(compressed_pixels, axis=0, return_counts=True)

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.bar(range(len(unique_colors)), counts, color=unique_colors / 255)
    ax2.set_title(f"Color Distribution (Exactly {k} Peaks)")
    ax2.set_xlabel("Cluster Index")
    ax2.set_ylabel("Pixel Count")
    st.pyplot(fig2)

    st.caption(
        "This histogram shows exactly **K peaks** → proof that only **K colors** remain after compression."
    )

    # Download button for compressed image
    with open("compressed_image.jpg", "rb") as file:
        st.download_button("📥 Download Compressed Image", file, file_name="compressed_image.jpg")
