import cv2
import numpy as np
import sys
import os


def peak_vector_rmse(original: np.ndarray, output: np.ndarray, peak: float = 255.0) -> float:
    """Compute Peak Vector RMSE between original and output images."""
    if original.shape != output.shape:
        raise ValueError(
            f"Image shapes must match: original {original.shape} vs output {output.shape}"
        )

    orig = np.asarray(original, dtype=np.float64)
    out = np.asarray(output, dtype=np.float64)

    # Vector RMSE: sqrt(mean((orig - out)^2)) over all pixels and channels
    mse = np.mean((orig - out) ** 2)
    rmse = np.sqrt(mse)

    return rmse / peak


def main():
    if len(sys.argv) != 3:
        print("Usage: python P-RMSE.py <original_image> <output_image>")
        sys.exit(1)

    orig_path = sys.argv[1]
    out_path = sys.argv[2]

    if not os.path.isfile(orig_path):
        print(f"Error: original image not found: {orig_path}")
        sys.exit(1)
    if not os.path.isfile(out_path):
        print(f"Error: output image not found: {out_path}")
        sys.exit(1)

    original = cv2.imread(orig_path)
    output = cv2.imread(out_path)

    if original is None:
        print(f"Error: could not load original image: {orig_path}")
        sys.exit(1)
    if output is None:
        print(f"Error: could not load output image: {out_path}")
        sys.exit(1)

    try:
        pv_rmse = peak_vector_rmse(original, output)
        print(f"Peak Vector RMSE: {pv_rmse:.6f}")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
