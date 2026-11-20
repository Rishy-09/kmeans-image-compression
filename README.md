# 🎨 K-Means Image Compression

This notebook shows how **K-Means Clustering** can compress an image by reducing its color palette, keeping visual quality intact while minimizing file size.

---

## 🧠 Why This Matters

Traditional compression methods like JPEG use **PCA** or **DCT** to eliminate spatial redundancy.

K-Means takes a different route — it compresses the image by reducing **color complexity**, clustering similar RGB values.

---

### 🔍 What it does:
- Applies unsupervised clustering (K-Means) to pixel RGB values
- Compresses an image by reducing thousands of colors to just 16
- Visual comparison of **original vs compressed** image

  
## 📊 Method Comparison

| Method         | Type           | What it reduces     | ML-based? | Output size | Info loss |
|----------------|----------------|----------------------|------------|--------------|------------|
| JPEG           | Transform-based| Frequency Redundancy | ❌         | High         | Medium     |
| PNG            | Lossless       | Redundant Bytes      | ❌         | Low          | None       |
| PCA            | Linear algebra | Feature Space         | ⚠️ Semi     | Low          | Some       |
| **K-Means**    | Clustering     | Pixel Color Values    | ✅         | Medium       | Low        |

---

### 📊 Results
![result](output_comparison.png)

## Quantitative Benchmark

To measure reconstruction quality beyond visual comparison, the implementation was benchmarked across multiple palette sizes using **Mean Squared Error (MSE)** and **Peak Signal-to-Noise Ratio (PSNR)**.

```bash
python benchmark.py original.png
```

| Colors (K) | Iterations | MSE | PSNR (dB) |
|---:|---:|---:|---:|
| 2 | 11 | 0.023206 | 16.34 |
| 4 | 35 | 0.009158 | 20.38 |
| 8 | 50 | 0.004896 | 23.10 |
| 16 | 50 | 0.002459 | 26.09 |
| 32 | 50 | 0.001416 | 28.49 |
| 64 | 50 | 0.000863 | 30.64 |

### Observation

Increasing `K` provides a larger representative color palette and improves reconstruction quality:

- **K = 2–4:** aggressive color quantization with substantial visual information loss.
- **K = 8–16:** stronger balance between palette reduction and image quality.
- **K = 32–64:** higher-fidelity reconstruction with progressively lower MSE and higher PSNR.

The benchmark measures **color-quantization quality**, not guaranteed encoded file-size reduction. Actual PNG/JPEG file size also depends on the image format and encoder.

### 🚀 Run it
- Clone this repo or open the notebook in [Google Colab](https://colab.research.google.com/)
- Try adjusting the number of clusters `K` to see quality vs. compression

---

Built using concepts from Andrew Ng’s Machine Learning Specialization.

