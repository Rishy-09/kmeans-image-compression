import numpy as np

from kmeans_compression import (
    initialize_centroids,
    find_closest_centroids,
    compute_centroids,
    run_kmeans,
    compress_image,
    mse,
    psnr,
)


def test_initialize_centroids_shape():
    pixels = np.random.default_rng(42).random((100, 3))
    rng = np.random.default_rng(42)

    centroids = initialize_centroids(pixels, 4, rng)

    assert centroids.shape == (4, 3)


def test_find_closest_centroids():
    pixels = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
    ])

    centroids = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
    ])

    assignments = find_closest_centroids(pixels, centroids)

    assert np.array_equal(assignments, [0, 1])


def test_compute_centroids():
    pixels = np.array([
        [0.0, 0.0, 0.0],
        [0.2, 0.2, 0.2],
        [0.8, 0.8, 0.8],
        [1.0, 1.0, 1.0],
    ])

    assignments = np.array([0, 0, 1, 1])
    old_centroids = np.zeros((2, 3))
    rng = np.random.default_rng(42)

    centroids = compute_centroids(
        pixels,
        assignments,
        old_centroids,
        rng,
    )

    expected = np.array([
        [0.1, 0.1, 0.1],
        [0.9, 0.9, 0.9],
    ])

    assert np.allclose(centroids, expected)


def test_kmeans_is_reproducible():
    pixels = np.random.default_rng(1).random((100, 3))

    centroids_a, assignments_a, _ = run_kmeans(
        pixels, k=4, seed=42
    )

    centroids_b, assignments_b, _ = run_kmeans(
        pixels, k=4, seed=42
    )

    assert np.allclose(centroids_a, centroids_b)
    assert np.array_equal(assignments_a, assignments_b)


def test_compressed_image_shape_and_range():
    image = np.random.default_rng(42).random((10, 10, 3))

    compressed, _ = compress_image(
        image,
        k=4,
        max_iters=10,
        seed=42,
    )

    assert compressed.shape == image.shape
    assert compressed.min() >= 0.0
    assert compressed.max() <= 1.0


def test_identical_images_have_zero_mse_and_infinite_psnr():
    image = np.ones((4, 4, 3))

    assert mse(image, image) == 0.0
    assert psnr(image, image) == float("inf")
