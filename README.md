# K-Means Image Compression

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![NumPy](https://img.shields.io/badge/NumPy-From%20Scratch-013243?logo=numpy)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter)
![Algorithm](https://img.shields.io/badge/Algorithm-K--Means-purple)
![License](https://img.shields.io/badge/License-MIT-green)
[![CI](https://github.com/Rishy-09/kmeans-image-compression/actions/workflows/ci.yml/badge.svg)](https://github.com/Rishy-09/kmeans-image-compression/actions/workflows/ci.yml)

A from-scratch implementation of **K-Means clustering for image color quantization**, built with NumPy.

The project treats every RGB pixel as a point in 3D color space and groups similar colors into `K` clusters. Each pixel is then replaced by its cluster centroid, reducing the image to a compact representative color palette.

<p align="center">
  <img src="output_comparison.png" width="850" alt="Original and K-Means compressed image comparison">
</p>

---

## How It Works

An RGB image can be represented as a collection of pixels:

```text
Image (H × W × 3)
        ↓
Flatten RGB pixels
        ↓
N × 3 color vectors
        ↓
Initialize K centroids
        ↓
Assign each pixel to nearest centroid
        ↓
Recompute cluster centroids
        ↓
Repeat until convergence / iteration limit
        ↓
Replace pixels with centroid colors
        ↓
Reconstruct compressed image
```

Instead of storing the full range of original colors, the reconstructed image uses only `K` representative colors.

---

## From-Scratch K-Means

The clustering algorithm is implemented directly with **NumPy** rather than using `sklearn.cluster.KMeans`.

The implementation includes:

- Random centroid initialization from image pixels
- Euclidean-distance based cluster assignment
- Centroid recomputation
- Empty-cluster handling
- Convergence detection
- Configurable maximum iterations
- Deterministic random seeding
- Vectorized nearest-centroid computation

The original educational implementation is preserved in:

```text
kmeans_image_compression.ipynb
```

A reusable command-line implementation is provided in:

```text
kmeans_compression.py
```

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/Rishy-09/kmeans-image-compression.git
cd kmeans-image-compression
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Compress an image

```bash
python kmeans_compression.py original.png --colors 16
```

By default, the compressed image is saved as:

```text
compressed.png
```

---

## CLI Usage

Choose the number of colors:

```bash
python kmeans_compression.py original.png --colors 8
```

Specify an output file:

```bash
python kmeans_compression.py original.png --colors 32 --output compressed_32.png
```

Control the maximum number of iterations:

```bash
python kmeans_compression.py original.png --colors 16 --max-iters 100
```

Use a different random seed:

```bash
python kmeans_compression.py original.png --colors 16 --seed 123
```

Example output:

```text
Input:       original.png
Output:      compressed.png
Colors:      16
Iterations:  50
MSE:         0.002459
PSNR:        26.09 dB
```

---

## Quantitative Benchmark

The implementation was benchmarked on the included sample image across several palette sizes.

Run the benchmark with:

```bash
python benchmark.py original.png
```

### Results

| Colors (K) | Iterations | MSE | PSNR (dB) |
|---:|---:|---:|---:|
| 2 | 11 | 0.023206 | 16.34 |
| 4 | 35 | 0.009158 | 20.38 |
| 8 | 50 | 0.004896 | 23.10 |
| 16 | 50 | 0.002459 | 26.09 |
| 32 | 50 | 0.001416 | 28.49 |
| 64 | 50 | 0.000863 | 30.64 |

### Interpretation

As `K` increases, the model has more representative colors available to reconstruct the image.

- **K = 2–4** — aggressive quantization with substantial visual information loss
- **K = 8–16** — stronger balance between palette reduction and reconstruction quality
- **K = 32–64** — increasingly high-fidelity reconstruction

Across this experiment:

```text
MSE
0.023206  →  0.000863

PSNR
16.34 dB  →  30.64 dB
```

when increasing the palette from **2 to 64 colors**.

For this run, `K = 2` and `K = 4` converged before the iteration limit. The experiments with `K ≥ 8` reached the configured maximum of 50 iterations, so those runs should not necessarily be interpreted as fully converged solutions.

---

## MSE and PSNR

### Mean Squared Error

MSE measures the average squared difference between the original and reconstructed pixel values.

```text
Lower MSE = lower reconstruction error
```

### Peak Signal-to-Noise Ratio

PSNR expresses reconstruction quality on a logarithmic scale.

```text
Higher PSNR = closer reconstruction to the original image
```

These metrics allow different palette sizes to be compared quantitatively rather than relying only on visual inspection.

---

## Color Quantization vs File Compression

This project primarily demonstrates **K-Means color quantization**.

K-Means reduces the number of distinct colors required to represent an image:

```text
Millions of possible RGB colors
              ↓
        K representative colors
```

This reduces palette complexity and introduces a controllable quality–representation tradeoff.

However, a smaller color palette does **not automatically guarantee a smaller PNG or JPEG file**.

Actual encoded file size also depends on factors such as:

- Image format
- Compression algorithm
- Encoder settings
- Image dimensions
- Spatial structure of the image
- Metadata

Therefore, the benchmark focuses on **reconstruction quality (MSE and PSNR)** rather than claiming a guaranteed file-size reduction.

---

## Project Structure

```text
kmeans-image-compression/
│
├── kmeans_compression.py
│   └── Reusable K-Means compression implementation and CLI
│
├── benchmark.py
│   └── Benchmarks multiple palette sizes using MSE and PSNR
│
├── kmeans_image_compression.ipynb
│   └── Educational notebook showing the algorithm step by step
│
├── original.png
│   └── Sample input image
│
├── output_comparison.png
│   └── Visual comparison of original and quantized images
│
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## Technical Highlights

- K-Means implemented from scratch
- NumPy-based vectorized distance computation
- RGB color-space clustering
- Configurable palette size
- Convergence-based stopping
- Empty-cluster recovery
- Deterministic experiments through random seeds
- Command-line interface
- MSE and PSNR evaluation
- Multi-`K` benchmarking
- Educational Jupyter notebook

---

## Limitations

### RGB is not perceptually uniform

Euclidean distance between two RGB values does not perfectly correspond to how different those colors appear to humans.

Clustering in a perceptually oriented color space such as **CIELAB** could produce different visual results.

### Computational cost

Standard K-Means repeatedly compares pixels against every centroid.

For large images or large values of `K`, this can become computationally expensive.

### Initialization affects the result

K-Means can converge to different local solutions depending on its initial centroids.

The CLI therefore exposes a random seed for reproducibility.

### PSNR is not perceptual quality

Higher PSNR generally indicates lower reconstruction error, but it does not perfectly represent human perception of image quality.

---

## Possible Improvements

Future extensions could include:

- K-Means++ centroid initialization
- CIELAB color-space clustering
- SSIM image-quality evaluation
- Pixel sampling for faster clustering
- Automatic convergence analysis
- Benchmark runtime measurements
- Automatic comparison images across multiple `K` values
- Elbow-method analysis for palette selection
- Unit tests and continuous integration

---

## Why This Project?

Image compression provides an intuitive way to understand unsupervised learning.

Instead of viewing K-Means only as points on a scatter plot, every cluster has an immediately visible interpretation:

```text
cluster centroid = representative color
```

Changing `K` makes the tradeoff between model complexity and reconstruction quality directly observable.

The project therefore connects:

**clustering → vector quantization → image processing → quantitative evaluation**

---

## License

This project is licensed under the **MIT License**. See [`LICENSE`](LICENSE) for details.
