#!/usr/bin/env python3
"""Download MNIST and save it in the format used by main2.py.

Output files (all next to this script, in ./data/mnist by default):
    trainX.npy  float32 (60000, 784)  values in [0, 1]
    trainY.npy  float32 (60000, 10)   one-hot labels
    testX.npy   float32 (10000, 784)  values in [0, 1]
    testY.npy   float32 (10000, 10)   one-hot labels

Usage:
    uv run --no-project --python /path/to/python download_mnist.py
    uv run --no-project --python /path/to/python download_mnist.py --out-dir data/mnist
"""

from __future__ import annotations

import argparse
import gzip
import struct
import sys
import urllib.request
from pathlib import Path

import numpy as np

DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "data" / "mnist"

# Mirrors are tried in order.
MIRRORS = (
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
)

ARCHIVES = {
    "train-images-idx3-ubyte.gz": "trainX.npy",
    "train-labels-idx1-ubyte.gz": "trainY.npy",
    "t10k-images-idx3-ubyte.gz": "testX.npy",
    "t10k-labels-idx1-ubyte.gz": "testY.npy",
}

REQUIRED = tuple(ARCHIVES.values())


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "mnist-downloader/1.0"})
    with urllib.request.urlopen(req, timeout=120) as response, open(tmp, "wb") as f:
        total = int(response.headers.get("Content-Length", 0))
        done = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {dest.name}: {done / total * 100:5.1f}%", end="", flush=True)
    print()
    tmp.replace(dest)


def _fetch(name: str, raw_dir: Path) -> Path:
    dest = raw_dir / name
    if dest.exists() and dest.stat().st_size > 0:
        return dest

    last_error = None
    for base in MIRRORS:
        url = base + name
        try:
            print(f"Downloading {url}")
            _download(url, dest)
            return dest
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(f"  failed: {exc}", file=sys.stderr)

    raise RuntimeError(f"Could not download {name}: {last_error}")


def _read_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"{path}: bad image magic {magic}, expected 2051")
        data = np.frombuffer(f.read(), dtype=np.uint8)

    expected = n * rows * cols
    if data.size != expected:
        raise ValueError(f"{path}: expected {expected} bytes, got {data.size}")
    return data.reshape(n, rows * cols)


def _read_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"{path}: bad label magic {magic}, expected 2049")
        data = np.frombuffer(f.read(), dtype=np.uint8)

    if data.size != n:
        raise ValueError(f"{path}: expected {n} labels, got {data.size}")
    return data


def _one_hot(labels: np.ndarray, n_classes: int = 10) -> np.ndarray:
    out = np.zeros((labels.size, n_classes), dtype=np.float32)
    out[np.arange(labels.size), labels] = 1.0
    return out


def ensure_mnist(out_dir: str | Path = DEFAULT_OUT_DIR, force: bool = False) -> Path:
    """Download/convert MNIST if needed and return the output directory."""
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not force and all((out_dir / name).exists() for name in REQUIRED):
        print(f"MNIST already present in {out_dir}")
        return out_dir

    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    paths = {name: _fetch(name, raw_dir) for name in ARCHIVES}

    print("Converting to .npy ...")
    train_x = _read_images(paths["train-images-idx3-ubyte.gz"])
    train_y = _read_labels(paths["train-labels-idx1-ubyte.gz"])
    test_x = _read_images(paths["t10k-images-idx3-ubyte.gz"])
    test_y = _read_labels(paths["t10k-labels-idx1-ubyte.gz"])

    np.save(out_dir / "trainX.npy", train_x.astype(np.float32) / 255.0)
    np.save(out_dir / "trainY.npy", _one_hot(train_y))
    np.save(out_dir / "testX.npy", test_x.astype(np.float32) / 255.0)
    np.save(out_dir / "testY.npy", _one_hot(test_y))

    print(f"Saved MNIST to: {out_dir}")
    print(f"  trainX: {train_x.shape}, trainY: {train_y.shape}")
    print(f"  testX:  {test_x.shape}, testY:  {test_y.shape}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Download MNIST for main2.py")
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help=f"output directory (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument("--force", action="store_true", help="re-download and re-convert")
    args = parser.parse_args()
    ensure_mnist(args.out_dir, force=args.force)


if __name__ == "__main__":
    main()
