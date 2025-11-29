Image Compression Using K-Means Clustering
Introduction

Image compression plays a crucial role in reducing storage requirements and improving transmission speed over the internet. With the growing number of high-resolution images, efficient compression techniques are necessary to save space without significantly affecting visual quality.
This project implements K-Means clustering for image compression by grouping similar colors together and replacing them with representative centroid values. The result is a compressed image that occupies less space while maintaining acceptable visual clarity.

How K-Means Clustering Works in Image Compression

K-Means clustering divides image pixel data into a predefined number of clusters (K) based on similarity between color values. Each pixel is assigned to the nearest centroid, and its color is replaced with that centroid’s color, reducing the overall number of unique colors in the image.

Detailed Algorithm Steps
1. Image Preprocessing

The input image is loaded and converted into a NumPy array.

Each pixel is represented as an RGB vector: (R, G, B).

2. Initialization

Select K random centroids from the image data (or use K-Means++ initialization).

3. Cluster Assignment

Each pixel is assigned to the nearest centroid using Euclidean Distance.

4. Centroid Update

The centroid value is recalculated as the mean of all pixels in the cluster.

5. Repeat

Steps 3 and 4 are repeated until centroids stop changing or a maximum iteration is reached.

6. Reconstruction

Each pixel is replaced with its centroid color.

The compressed image is generated.

Key Parameters in K-Means Compression
Parameter	Description
K	Number of clusters (controls compression level)
Initialization	Random or K-Means++
Distance Metric	Euclidean distance
Iterations	Repeated until convergence
Advantages of This Approach

Easy to implement and understand.

Highly flexible compression through adjustable K value.

Significant size reduction possible.

Works without labeled data (unsupervised learning).

Visual proof through centroid grouping.

Efficient for many types of images.

Limitations

Requires K to be chosen manually.

Not as smooth as JPEG for natural photos.

Sensitive to centroid initialization.

Slower than traditional compression for large images.

Applications

Image size reduction in web platforms.

Storage optimization systems.

Data preprocessing for computer vision.

Image segmentation tasks.

Academic learning for machine learning concepts.

Dataset

This project does not require a fixed dataset.
Each image itself serves as the dataset since pixel values act as data points.

For testing, standard images such as:

Peppers

Lenna

Landscape

CIFAR-10 images
can be used.
KMeans_Image_Compression/
   ├── imageapp.py
   ├── proof.py
   ├── requirements.txt
   ├── sample_images/
   ├── README.md
   └── results/
Getting Started
Requirements

Install Python 3.8+ and required packages:

pip install numpy pillow scikit-learn streamlit opencv-python


Or use:

pip install -r requirements.txt

Running the Application

To launch the UI:

streamlit run imageapp.py


or

streamlit run proof.py


Upload an image and select compression level:

Large (High compression)

Medium

Small (Best quality)

Output

Displays original and compressed images.

Shows file size reduction.

Displays centroids for proof.

Visualizes color clusters.

Proof of K-Means Compression (VCSS & Centroids)

Centroid values show actual clustering.

WCSS (Within-Cluster Sum of Squares) proves optimization.

Palette images display reduced dominant colors.

Cluster overlays visualize pixel grouping.
