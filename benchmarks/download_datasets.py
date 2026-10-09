import os
import sys
import hashlib
import urllib.request
import numpy as np
import h5py

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CHECKSUMS_FILE = os.path.join(DATA_DIR, "CHECKSUMS.txt")

DATASET_URLS = {
    "fashion-mnist-784-euclidean": "http://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5",
    "sift-128-euclidean": "http://ann-benchmarks.com/sift-128-euclidean.hdf5",
    "glove-100-angular": "http://ann-benchmarks.com/glove-100-angular.hdf5",
}


def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


def compute_sha256(filepath: str) -> str:
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            sha256.update(chunk)
    return sha256.hexdigest()


def update_checksums_file(filename: str, sha256_hash: str):
    ensure_dir(DATA_DIR)
    entries = {}
    if os.path.exists(CHECKSUMS_FILE):
        with open(CHECKSUMS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    entries[parts[1]] = parts[0]
    entries[filename] = sha256_hash
    with open(CHECKSUMS_FILE, "w", encoding="utf-8") as f:
        for fname, h in sorted(entries.items()):
            f.write(f"{h}  {fname}\n")


def generate_synthetic_dataset(n_train=10000, n_test=1000, dim=64, n_clusters=20, metric="l2", seed=42):
    """Explicit synthetic clustered dataset generator for smoke and stress tests."""
    rng = np.random.default_rng(seed)
    centroids = rng.standard_normal((n_clusters, dim)).astype(np.float32)

    # Train vectors
    train_labels = rng.integers(0, n_clusters, size=n_train)
    noise_train = rng.standard_normal((n_train, dim)).astype(np.float32) * 0.1
    train_data = (centroids[train_labels] + noise_train).astype(np.float32)

    # Test vectors
    test_labels = rng.integers(0, n_clusters, size=n_test)
    noise_test = rng.standard_normal((n_test, dim)).astype(np.float32) * 0.1
    test_data = (centroids[test_labels] + noise_test).astype(np.float32)

    if metric == "cosine":
        train_norms = np.linalg.norm(train_data, axis=1, keepdims=True)
        train_data = (train_data / np.maximum(train_norms, 1e-12)).astype(np.float32)
        test_norms = np.linalg.norm(test_data, axis=1, keepdims=True)
        test_data = (test_data / np.maximum(test_norms, 1e-12)).astype(np.float32)

    return train_data, test_data, metric


def download_ann_benchmark_dataset(name: str) -> str:
    ensure_dir(DATA_DIR)
    file_path = os.path.join(DATA_DIR, f"{name}.hdf5")

    if os.path.exists(file_path):
        print(f"Dataset already exists: {file_path}")
        return file_path

    url = DATASET_URLS.get(name, f"http://ann-benchmarks.com/{name}.hdf5")
    print(f"Downloading {name} from {url}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})

    try:
        with urllib.request.urlopen(req, timeout=120) as resp, open(file_path, "wb") as out_f:
            total_size = int(resp.headers.get("content-length", 0))
            downloaded = 0
            block_size = 1024 * 1024
            while True:
                chunk = resp.read(block_size)
                if not chunk:
                    break
                downloaded += len(chunk)
                out_f.write(chunk)
                if total_size > 0:
                    pct = (downloaded / total_size) * 100
                    mb = downloaded / (1024 * 1024)
                    tot_mb = total_size / (1024 * 1024)
                    print(f"\r  Progress: {mb:.1f} MB / {tot_mb:.1f} MB ({pct:.1f}%)", end="", flush=True)
            print("")

        print(f"Download complete: {file_path}")
        sha = compute_sha256(file_path)
        update_checksums_file(f"{name}.hdf5", sha)
        print(f"SHA-256: {sha}")
        return file_path
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise RuntimeError(f"Failed to download {name} from {url}: {e}")


def load_dataset(name: str):
    if name.startswith("synthetic"):
        # Explicit synthetic format: synthetic_<n>_<dim>_<metric>
        parts = name.split("_")
        n = int(parts[1]) if len(parts) > 1 else 10000
        dim = int(parts[2]) if len(parts) > 2 else 64
        metric = parts[3] if len(parts) > 3 else "l2"
        train, test, m = generate_synthetic_dataset(n_train=n, n_test=min(1000, n // 5), dim=dim, metric=metric)
        return train, test, None, None, m

    # Download real dataset
    path = download_ann_benchmark_dataset(name)
    with h5py.File(path, "r") as f:
        train = np.array(f["train"], dtype=np.float32)
        test = np.array(f["test"], dtype=np.float32)
        neighbors = np.array(f["neighbors"]) if "neighbors" in f else None
        distances = np.array(f["distances"], dtype=np.float32) if "distances" in f else None
        metric_attr = f.attrs.get("distance", "euclidean")
        metric_map = {"euclidean": "l2", "angular": "cosine", "dot": "ip"}
        metric = metric_map.get(str(metric_attr).lower(), "l2")
        return train, test, neighbors, distances, metric


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download and verify ANN benchmark datasets")
    parser.add_argument("dataset", nargs="?", default="fashion-mnist-784-euclidean", help="Dataset name")
    parser.add_argument("--all", action="store_true", help="Download all standard datasets")
    args = parser.parse_args()

    if args.all:
        for ds in ["fashion-mnist-784-euclidean", "sift-128-euclidean", "glove-100-angular"]:
            try:
                p = download_ann_benchmark_dataset(ds)
                print(f"Verified: {p}")
            except Exception as ex:
                print(f"Could not download {ds}: {ex}")
    else:
        tr, te, nbrs, dists, m = load_dataset(args.dataset)
        print(f"Dataset {args.dataset}: train={tr.shape}, test={te.shape}, metric={m}")
