#!/usr/bin/env python3
"""Download CIFAR-10 and save it for cifar10_backprop.py.

Output (default: ./data/cifar10 next to this file):
    trainX.npy  float32 (50000, 3, 32, 32)  values in [0, 1]
    trainY.npy  float32 (50000, 10)         one-hot labels
    testX.npy   float32 (10000, 3, 32, 32)  values in [0, 1]
    testY.npy   float32 (10000, 10)         one-hot labels

Usage:
    uv run --active python download_cifar10.py
    uv run --active python download_cifar10.py --out-dir /root/snn/data/cifar10
"""

from __future__ import annotations

import argparse
import pickle
import sys
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "data" / "cifar10"

URLS = (
    "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
    "https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz",
)

ARCHIVE = "cifar-10-python.tar.gz"
EXTRACTED = "cifar-10-batches-py"
REQUIRED = ("trainX.npy", "trainY.npy", "testX.npy", "testY.npy")


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "cifar10-downloader/1.0"})
    with urllib.request.urlopen(req, timeout=180) as response, open(tmp, "wb") as f:
        total = int(response.headers.get("Content-Length", 0))
        done = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                print(
                    f"\r  {ARCHIVE}: {done / total * 100:5.1f}%",
                    end="",
                    flush=True,
                )
    print()
    tmp.replace(dest)


def _fetch(raw_dir: Path) -> Path:
    dest = raw_dir / ARCHIVE
    if dest.exists() and dest.stat().st_size > 0:
        return dest

    last_error = None
    for url in URLS:
        try:
            print(f"Downloading {url}")
            _download(url, dest)
            return dest
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(f"  failed: {exc}", file=sys.stderr)

    raise RuntimeError(f"Could not download CIFAR-10: {last_error}")


def _unpickle(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f, encoding="bytes")


def _one_hot(labels: np.ndarray, n_classes: int = 10) -> np.ndarray:
    out = np.zeros((labels.size, n_classes), dtype=np.float32)
    out[np.arange(labels.size), labels] = 1.0
    return out


def ensure_cifar10(out_dir: str | Path = DEFAULT_OUT_DIR, force: bool = False) -> Path:
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not force and all((out_dir / name).exists() for name in REQUIRED):
        print(f"CIFAR-10 already present in {out_dir}")
        return out_dir

    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = _fetch(raw_dir)

    extract_dir = raw_dir / EXTRACTED
    if force or not extract_dir.exists():
        print("Extracting ...")
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(raw_dir)

    batch_dir = raw_dir / EXTRACTED
    train_data = []
    train_labels = []
    for i in range(1, 6):
        batch = _unpickle(batch_dir / f"data_batch_{i}")
        train_data.append(batch[b"data"])
        train_labels.extend(batch[b"labels"])

    train_x = np.concatenate(train_data, axis=0).reshape(-1, 3, 32, 32)
    train_y = np.asarray(train_labels, dtype=np.int64)

    test = _unpickle(batch_dir / "test_batch")
    test_x = test[b"data"].reshape(-1, 3, 32, 32)
    test_y = np.asarray(test[b"labels"], dtype=np.int64)

    np.save(out_dir / "trainX.npy", train_x.astype(np.float32) / 255.0)
    np.save(out_dir / "trainY.npy", _one_hot(train_y))
    np.save(out_dir / "testX.npy", test_x.astype(np.float32) / 255.0)
    np.save(out_dir / "testY.npy", _one_hot(test_y))

    print(f"Saved CIFAR-10 to: {out_dir}")
    print(f"  trainX: {train_x.shape}, trainY: {train_y.shape}")
    print(f"  testX:  {test_x.shape}, testY:  {test_y.shape}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Download CIFAR-10 for cifar10_backprop.py")
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help=f"output directory (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument("--force", action="store_true", help="re-download and re-convert")
    args = parser.parse_args()
    ensure_cifar10(args.out_dir, force=args.force)


if __name__ == "__main__":
    main()
