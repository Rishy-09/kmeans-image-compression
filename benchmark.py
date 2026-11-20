import argparse

from kmeans_compression import load_image, compress_image, mse, psnr


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark K-Means image compression across palette sizes."
    )
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument(
        "--colors",
        nargs="+",
        type=int,
        default=[2, 4, 8, 16, 32, 64],
        help="Palette sizes to benchmark",
    )
    parser.add_argument("--max-iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    image = load_image(args.image)

    print()
    print("K-Means Image Compression Benchmark")
    print("-" * 58)
    print(f"{'Colors':>8} {'Iterations':>12} {'MSE':>14} {'PSNR (dB)':>14}")
    print("-" * 58)

    for k in args.colors:
        compressed, iterations = compress_image(
            image,
            k=k,
            max_iters=args.max_iters,
            seed=args.seed,
        )

        error = mse(image, compressed)
        quality = psnr(image, compressed)

        print(
            f"{k:>8} "
            f"{iterations:>12} "
            f"{error:>14.6f} "
            f"{quality:>14.2f}"
        )


if __name__ == "__main__":
    main()
