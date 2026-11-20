import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_image(path):
    """Load an RGB image and normalize pixel values to [0, 1]."""
    image = plt.imread(path)

    if image.ndim != 3:
        raise ValueError("Expected an RGB/RGBA image.")

    # Remove alpha channel if present
    if image.shape[2] == 4:
        image = image[:, :, :3]

    # Integer images usually use [0, 255]
    if np.issubdtype(image.dtype, np.integer):
        image = image.astype(np.float32) / 255.0
    else:
        image = image.astype(np.float32)
        if image.max() > 1.0:
            image /= 255.0

    return np.clip(image, 0.0, 1.0)


def initialize_centroids(pixels, k, rng):
    """Initialize centroids using randomly selected image pixels."""
    if k > len(pixels):
        raise ValueError("Number of colors cannot exceed number of pixels.")

    indices = rng.choice(len(pixels), size=k, replace=False)
    return pixels[indices].copy()


def find_closest_centroids(pixels, centroids):
    """Assign every pixel to its nearest centroid."""
    distances = np.sum(
        (pixels[:, None, :] - centroids[None, :, :]) ** 2,
        axis=2,
    )
    return np.argmin(distances, axis=1)


def compute_centroids(pixels, assignments, centroids, rng):
    """Recompute cluster centroids while handling empty clusters."""
    k = len(centroids)
    new_centroids = np.empty_like(centroids)

    for cluster in range(k):
        members = pixels[assignments == cluster]

        if len(members) == 0:
            # Reinitialize an empty cluster using an existing pixel.
            new_centroids[cluster] = pixels[rng.integers(len(pixels))]
        else:
            new_centroids[cluster] = members.mean(axis=0)

    return new_centroids


def run_kmeans(pixels, k, max_iters=50, tolerance=1e-5, seed=42):
    """Run K-Means until convergence or max_iters is reached."""
    rng = np.random.default_rng(seed)
    centroids = initialize_centroids(pixels, k, rng)

    for iteration in range(1, max_iters + 1):
        assignments = find_closest_centroids(pixels, centroids)
        new_centroids = compute_centroids(
            pixels, assignments, centroids, rng
        )

        shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids

        if shift < tolerance:
            break

    assignments = find_closest_centroids(pixels, centroids)

    return centroids, assignments, iteration


def compress_image(image, k=16, max_iters=50, seed=42):
    """Compress an image by reducing it to K representative colors."""
    height, width, channels = image.shape
    pixels = image.reshape(-1, channels)

    centroids, assignments, iterations = run_kmeans(
        pixels,
        k=k,
        max_iters=max_iters,
        seed=seed,
    )

    compressed = centroids[assignments].reshape(height, width, channels)

    return np.clip(compressed, 0.0, 1.0), iterations


def mse(original, compressed):
    return float(np.mean((original - compressed) ** 2))


def psnr(original, compressed):
    error = mse(original, compressed)

    if error == 0:
        return float("inf")

    return float(10 * np.log10(1.0 / error))


def save_image(image, output_path):
    plt.imsave(output_path, image)


def main():
    parser = argparse.ArgumentParser(
        description="Compress an image using K-Means color quantization."
    )

    parser.add_argument("image", help="Path to the input image")

    parser.add_argument(
        "-k",
        "--colors",
        type=int,
        default=16,
        help="Number of colors in the compressed palette (default: 16)",
    )

    parser.add_argument(
        "-o",
        "--output",
        default="compressed.png",
        help="Output image path (default: compressed.png)",
    )

    parser.add_argument(
        "--max-iters",
        type=int,
        default=50,
        help="Maximum K-Means iterations (default: 50)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    if args.colors < 1:
        parser.error("--colors must be at least 1")

    image_path = Path(args.image)

    if not image_path.exists():
        parser.error(f"Image not found: {image_path}")

    image = load_image(image_path)

    compressed, iterations = compress_image(
        image,
        k=args.colors,
        max_iters=args.max_iters,
        seed=args.seed,
    )

    save_image(compressed, args.output)

    print(f"Input:       {image_path}")
    print(f"Output:      {args.output}")
    print(f"Colors:      {args.colors}")
    print(f"Iterations:  {iterations}")
    print(f"MSE:         {mse(image, compressed):.6f}")
    print(f"PSNR:        {psnr(image, compressed):.2f} dB")


if __name__ == "__main__":
    main()
