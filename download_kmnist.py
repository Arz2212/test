#!/usr/bin/env python3
"""Download KMNIST (Kuzushiji-MNIST) for kmnist_solver.py.

Output (default: ./data/kmnist next to this file):
    trainX.npy  float32 (60000, 784)  values in [0, 1]
    trainY.npy  float32 (60000, 10)   one-hot labels
    testX.npy   float32 (10000, 784)  values in [0, 1]
    testY.npy   float32 (10000, 10)   one-hot labels

Usage:
    uv run --active python download_kmnist.py
    uv run --active python download_kmnist.py --out-dir /root/snn/data/kmnist
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import numpy as np

DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "data" / "kmnist"

MIRRORS = (
    "https://codh.rois.ac.jp/kmnist/dataset/kmnist/",
    "http://codh.rois.ac.jp/kmnist/dataset/kmnist/",
)

ARCHIVES = (
    "kmnist-train-imgs.npz",
    "kmnist-train-labels.npz",
    "kmnist-test-imgs.npz",
    "kmnist-test-labels.npz",
)

REQUIRED = ("trainX.npy", "trainY.npy", "testX.npy", "testY.npy")


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as response, open(tmp, "wb") as f:
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
    import time
    dest = raw_dir / name
    if dest.exists() and dest.stat().st_size > 0:
        return dest

    last_error = None
    for attempt in range(1, 5):
        for base in MIRRORS:
            url = base + name
            try:
                print(f"Downloading {url} (attempt {attempt})", flush=True)
                _download(url, dest)
                time.sleep(0.5)
                return dest
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                print(f"  failed: {exc}", file=sys.stderr, flush=True)
                time.sleep(2 * attempt)

    raise RuntimeError(f"Could not download {name}: {last_error}")


def _read_npz(path: Path) -> np.ndarray:
    with np.load(path) as z:
        key = "arr_0" if "arr_0" in z.files else z.files[0]
        return z[key]


def _one_hot(labels: np.ndarray, n_classes: int = 10) -> np.ndarray:
    out = np.zeros((labels.size, n_classes), dtype=np.float32)
    out[np.arange(labels.size), labels] = 1.0
    return out


def ensure_kmnist(out_dir: str | Path = DEFAULT_OUT_DIR, force: bool = False) -> Path:
    """Download/convert KMNIST if needed and return the output directory."""
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not force and all((out_dir / name).exists() for name in REQUIRED):
        print(f"KMNIST already present in {out_dir}")
        return out_dir

    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: _fetch(name, raw_dir) for name in ARCHIVES}

    print("Converting to .npy ...")
    train_x = _read_npz(paths["kmnist-train-imgs.npz"]).reshape(-1, 28 * 28)
    train_y = _read_npz(paths["kmnist-train-labels.npz"]).reshape(-1)
    test_x = _read_npz(paths["kmnist-test-imgs.npz"]).reshape(-1, 28 * 28)
    test_y = _read_npz(paths["kmnist-test-labels.npz"]).reshape(-1)

    np.save(out_dir / "trainX.npy", train_x.astype(np.float32) / 255.0)
    np.save(out_dir / "trainY.npy", _one_hot(train_y))
    np.save(out_dir / "testX.npy", test_x.astype(np.float32) / 255.0)
    np.save(out_dir / "testY.npy", _one_hot(test_y))

    print(f"Saved KMNIST to: {out_dir}")
    print(f"  trainX: {train_x.shape}, trainY: {train_y.shape}")
    print(f"  testX:  {test_x.shape}, testY:  {test_y.shape}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Download KMNIST for kmnist_solver.py")
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help=f"output directory (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument("--force", action="store_true", help="re-download and re-convert")
    args = parser.parse_args()
    ensure_kmnist(args.out_dir, force=args.force)


if __name__ == "__main__":
    main()
