import json
import os
import time
from typing import Optional
import numpy as np
import typer
import uvicorn

from hnswdb import HNSWIndex, FlatIndex

app = typer.Typer(name="hnswdb", help="HNSWDB Approximate Nearest Neighbor Vector Database CLI")


def load_vectors(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        return np.load(path).astype(np.float32)
    elif path.endswith(".fvecs"):
        with open(path, "rb") as f:
            a = np.fromfile(f, dtype=np.int32)
            if len(a) == 0:
                return np.empty((0, 0), dtype=np.float32)
            d = a[0]
            return a.view(np.float32).reshape(-1, d + 1)[:, 1:].copy()
    else:
        # Fallback load text/csv or npy
        try:
            return np.load(path).astype(np.float32)
        except Exception:
            return np.loadtxt(path, delimiter=",").astype(np.float32)


@app.command()
def build(
    input: str = typer.Option(..., "--input", "-i", help="Input vector file (.npy or .fvecs)"),
    out: str = typer.Option(..., "--out", "-o", help="Output index file path (.hdb)"),
    dim: Optional[int] = typer.Option(None, "--dim", "-d", help="Vector dimension"),
    metric: str = typer.Option("l2", "--metric", "-m", help="Distance metric: l2, ip, cosine"),
    M: int = typer.Option(16, "--M", help="HNSW M parameter (2..128)"),
    ef_construction: int = typer.Option(200, "--ef-construction", help="ef_construction parameter"),
    seed: int = typer.Option(42, "--seed", help="Random seed for level generator"),
    threads: int = typer.Option(0, "--threads", "-t", help="Number of worker threads (0 = hardware concurrency)"),
):
    """Build an HNSW index from a vector dataset and save to disk."""
    typer.echo(f"Loading vectors from {input}...")
    vecs = load_vectors(input)
    n, d = vecs.shape
    if dim is not None and d != dim:
        raise typer.BadParameter(f"Vector dimension {d} does not match specified --dim {dim}")

    typer.echo(f"Building index: {n} vectors, dim={d}, metric={metric}, M={M}, ef_construction={ef_construction}...")
    t0 = time.perf_counter()

    idx = HNSWIndex(
        dim=d,
        metric=metric,
        max_elements=n,
        M=M,
        ef_construction=ef_construction,
        seed=seed,
    )
    ids = np.arange(n, dtype=np.int64)
    idx.add(vecs, ids, num_threads=threads)
    build_time = time.perf_counter() - t0

    typer.echo(f"Build completed in {build_time:.3f}s. Saving to {out}...")
    idx.save(out)
    typer.echo(f"Index successfully saved to {out}")


@app.command()
def query(
    index: str = typer.Option(..., "--index", help="Index file path (.hdb)"),
    query_file: str = typer.Option(..., "--query", help="Query vector file (.npy or .fvecs)"),
    k: int = typer.Option(10, "--k", "-k", help="Number of nearest neighbors to retrieve"),
    ef_search: Optional[int] = typer.Option(None, "--ef-search", help="Search expansion factor ef"),
    out: Optional[str] = typer.Option(None, "--out", "-o", help="Output results CSV path"),
    threads: int = typer.Option(0, "--threads", "-t", help="Search worker threads"),
):
    """Query an HNSW index with query vectors."""
    idx = HNSWIndex.load(index)
    queries = load_vectors(query_file)
    nq = queries.shape[0]

    t0 = time.perf_counter()
    res_ids, res_dists = idx.search(queries, k=k, ef_search=ef_search, num_threads=threads)
    took_ms = (time.perf_counter() - t0) * 1000.0

    typer.echo(f"Queried {nq} items for top-{k} in {took_ms:.2f}ms (avg {took_ms/nq:.3f}ms/query)")

    if out:
        with open(out, "w", encoding="utf-8") as f:
            f.write("query_idx,rank,id,distance\n")
            for q_i in range(nq):
                for rank in range(k):
                    f.write(f"{q_i},{rank},{res_ids[q_i, rank]},{res_dists[q_i, rank]:.6f}\n")
        typer.echo(f"Saved results to {out}")


@app.command()
def info(
    index: str = typer.Option(..., "--index", help="Index file path (.hdb)"),
):
    """Display detailed JSON statistics of an index."""
    idx = HNSWIndex.load(index)
    st = idx.stats()
    typer.echo(json.dumps(st, indent=2))


@app.command()
def bench(
    index: str = typer.Option(..., "--index", help="Index file path (.hdb)"),
    queries_file: str = typer.Option(..., "--queries", help="Query vector file (.npy or .fvecs)"),
    k: int = typer.Option(10, "--k", "-k", help="Top-k neighbors"),
    ef_search: int = typer.Option(50, "--ef-search", help="ef_search parameter"),
    threads: int = typer.Option(0, "--threads", "-t", help="Number of threads for batch QPS"),
    warmup: int = typer.Option(50, "--warmup", help="Warmup queries count"),
):
    """Run latency and throughput benchmarks against an existing index."""
    idx = HNSWIndex.load(index)
    queries = load_vectors(queries_file)
    nq = len(queries)

    # Warmup
    warmup_n = min(warmup, nq)
    idx.search(queries[:warmup_n], k=k, ef_search=ef_search)

    # Single-query latency measurement
    latencies = []
    for i in range(min(500, nq)):
        q = queries[i:i+1]
        t0 = time.perf_counter_ns()
        idx.search(q, k=k, ef_search=ef_search)
        latencies.append((time.perf_counter_ns() - t0) / 1e6)

    lat_arr = np.array(latencies)
    p50 = float(np.percentile(lat_arr, 50))
    p95 = float(np.percentile(lat_arr, 95))
    p99 = float(np.percentile(lat_arr, 99))

    # Multi-threaded batch QPS
    repeats = 3
    qps_runs = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        idx.search(queries, k=k, ef_search=ef_search, num_threads=threads)
        elapsed = time.perf_counter() - t0
        qps_runs.append(nq / elapsed)

    median_qps = float(np.median(qps_runs))

    results = {
        "queries_count": nq,
        "k": k,
        "ef_search": ef_search,
        "threads": threads,
        "qps": round(median_qps, 2),
        "latency_ms": {
            "mean": round(float(np.mean(lat_arr)), 3),
            "p50": round(p50, 3),
            "p95": round(p95, 3),
            "p99": round(p99, 3),
        }
    }
    typer.echo(json.dumps(results, indent=2))


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Host address"),
    port: int = typer.Option(8000, "--port", help="Port to listen on"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload"),
):
    """Start the HNSWDB REST API server."""
    typer.echo(f"Starting HNSWDB REST server at http://{host}:{port}...")
    uvicorn.run("hnswdb.server:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
