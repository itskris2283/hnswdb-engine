import os
import time
import tempfile
import psutil
import numpy as np

import hnswdb
try:
    import faiss
except ImportError:
    faiss = None

try:
    import hnswlib
except ImportError:
    hnswlib = None


class BaseEngine:
    def __init__(self, dim: int, metric: str = "l2", **kwargs):
        self.dim = dim
        self.metric = metric
        self.kwargs = kwargs
        self.build_time = 0.0
        self.rss_delta_mb = 0.0

    def build(self, data: np.ndarray, num_threads: int = 0) -> float:
        raise NotImplementedError

    def search(self, queries: np.ndarray, k: int = 10, ef_search: int = 50, num_threads: int = 0):
        raise NotImplementedError

    def get_memory_bytes(self) -> int:
        return 0

    def get_disk_bytes(self) -> int:
        return 0


class HNSWDBEngine(BaseEngine):
    def __init__(self, dim: int, metric: str = "l2", M: int = 16, ef_construction: int = 200, seed: int = 42, **kwargs):
        super().__init__(dim, metric, **kwargs)
        self.M = M
        self.ef_construction = ef_construction
        self.seed = seed
        self.index = None

    def build(self, data: np.ndarray, num_threads: int = 0) -> float:
        n = data.shape[0]
        data_c = np.ascontiguousarray(data, dtype=np.float32)
        proc = psutil.Process()
        rss_before = proc.memory_info().rss

        self.index = hnswdb.HNSWIndex(
            dim=self.dim,
            metric=self.metric,
            max_elements=n,
            M=self.M,
            ef_construction=self.ef_construction,
            seed=self.seed,
        )
        ids = np.arange(n, dtype=np.int64)
        t0 = time.perf_counter()
        self.index.add(data_c, ids, num_threads=num_threads)
        self.build_time = time.perf_counter() - t0

        rss_after = proc.memory_info().rss
        self.rss_delta_mb = max(0.0, (rss_after - rss_before) / (1024.0 * 1024.0))
        return self.build_time

    def search(self, queries: np.ndarray, k: int = 10, ef_search: int = 50, num_threads: int = 0):
        q_c = np.ascontiguousarray(queries, dtype=np.float32)
        return self.index.search(q_c, k=k, ef_search=ef_search, num_threads=num_threads)

    def bench_search_latency(self, queries: np.ndarray, k: int = 10, ef_search: int = 50) -> float:
        q_c = np.ascontiguousarray(queries, dtype=np.float32)
        return self.index.bench_search_latency(q_c, k=k, ef=ef_search)

    def get_memory_bytes(self) -> int:
        if self.index:
            return int(self.index.stats().get("memory_bytes", 0))
        return 0

    def get_disk_bytes(self) -> int:
        if not self.index:
            return 0
        tmp_path = os.path.join(tempfile.gettempdir(), f"hnswdb_tmp_{time.perf_counter_ns()}.hdb")
        try:
            self.index.save(tmp_path)
            return os.path.getsize(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass


class FaissHNSWEngine(BaseEngine):
    def __init__(self, dim: int, metric: str = "l2", M: int = 16, ef_construction: int = 200, **kwargs):
        super().__init__(dim, metric, **kwargs)
        if faiss is None:
            raise RuntimeError("FAISS is not installed")
        self.M = M
        self.ef_construction = ef_construction
        self.index = None

    def build(self, data: np.ndarray, num_threads: int = 0) -> float:
        th = num_threads if num_threads > 0 else (psutil.cpu_count(logical=True) or 4)
        faiss.omp_set_num_threads(th)

        proc = psutil.Process()
        rss_before = proc.memory_info().rss

        if self.metric == "l2":
            self.index = faiss.IndexHNSWFlat(self.dim, self.M, faiss.METRIC_L2)
        elif self.metric in ["ip", "cosine"]:
            self.index = faiss.IndexHNSWFlat(self.dim, self.M, faiss.METRIC_INNER_PRODUCT)
        else:
            raise ValueError(f"Unknown metric {self.metric}")

        self.index.hnsw.efConstruction = self.ef_construction

        train_data = np.ascontiguousarray(data, dtype=np.float32)
        if self.metric == "cosine":
            norms = np.linalg.norm(train_data, axis=1, keepdims=True)
            train_data = (train_data / np.maximum(norms, 1e-12)).astype(np.float32)

        t0 = time.perf_counter()
        self.index.add(train_data)
        self.build_time = time.perf_counter() - t0

        rss_after = proc.memory_info().rss
        self.rss_delta_mb = max(0.0, (rss_after - rss_before) / (1024.0 * 1024.0))
        return self.build_time

    def search(self, queries: np.ndarray, k: int = 10, ef_search: int = 50, num_threads: int = 0):
        th = num_threads if num_threads > 0 else 1
        faiss.omp_set_num_threads(th)
        self.index.hnsw.efSearch = ef_search

        q_data = np.ascontiguousarray(queries, dtype=np.float32)
        if self.metric == "cosine":
            norms = np.linalg.norm(q_data, axis=1, keepdims=True)
            q_data = (q_data / np.maximum(norms, 1e-12)).astype(np.float32)

        dists, ids = self.index.search(q_data, k)
        return ids.astype(np.int64), dists.astype(np.float32)

    def get_disk_bytes(self) -> int:
        if not self.index:
            return 0
        tmp_path = os.path.join(tempfile.gettempdir(), f"faiss_tmp_{time.perf_counter_ns()}.faiss")
        try:
            faiss.write_index(self.index, tmp_path)
            return os.path.getsize(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass


class HnswlibEngine(BaseEngine):
    def __init__(self, dim: int, metric: str = "l2", M: int = 16, ef_construction: int = 200, **kwargs):
        super().__init__(dim, metric, **kwargs)
        if hnswlib is None:
            raise RuntimeError("hnswlib is not installed")
        self.M = M
        self.ef_construction = ef_construction
        self.index = None

    def build(self, data: np.ndarray, num_threads: int = 0) -> float:
        space = "l2" if self.metric == "l2" else ("ip" if self.metric == "ip" else "cosine")
        n = data.shape[0]
        data_c = np.ascontiguousarray(data, dtype=np.float32)

        proc = psutil.Process()
        rss_before = proc.memory_info().rss

        self.index = hnswlib.Index(space=space, dim=self.dim)
        self.index.init_index(max_elements=n, ef_construction=self.ef_construction, M=self.M)

        threads = num_threads if num_threads > 0 else -1
        t0 = time.perf_counter()
        self.index.add_items(data_c, np.arange(n), num_threads=threads)
        self.build_time = time.perf_counter() - t0

        rss_after = proc.memory_info().rss
        self.rss_delta_mb = max(0.0, (rss_after - rss_before) / (1024.0 * 1024.0))
        return self.build_time

    def search(self, queries: np.ndarray, k: int = 10, ef_search: int = 50, num_threads: int = 0):
        self.index.set_ef(ef_search)
        threads = num_threads if num_threads > 0 else 1
        self.index.set_num_threads(threads)
        q_c = np.ascontiguousarray(queries, dtype=np.float32)
        ids, dists = self.index.knn_query(q_c, k=k, num_threads=threads)
        return ids.astype(np.int64), dists.astype(np.float32)

    def get_disk_bytes(self) -> int:
        if not self.index:
            return 0
        tmp_path = os.path.join(tempfile.gettempdir(), f"hnswlib_tmp_{time.perf_counter_ns()}.hnswlib")
        try:
            self.index.save_index(tmp_path)
            return os.path.getsize(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass


class FaissFlatEngine(BaseEngine):
    def __init__(self, dim: int, metric: str = "l2", **kwargs):
        super().__init__(dim, metric, **kwargs)
        self.index = None

    def build(self, data: np.ndarray, num_threads: int = 0) -> float:
        th = num_threads if num_threads > 0 else (psutil.cpu_count(logical=True) or 4)
        faiss.omp_set_num_threads(th)
        if self.metric == "l2":
            self.index = faiss.IndexFlatL2(self.dim)
        else:
            self.index = faiss.IndexFlatIP(self.dim)

        train_data = np.ascontiguousarray(data, dtype=np.float32)
        if self.metric == "cosine":
            norms = np.linalg.norm(train_data, axis=1, keepdims=True)
            train_data = (train_data / np.maximum(norms, 1e-12)).astype(np.float32)

        self.index.add(train_data)
        return 0.0

    def search(self, queries: np.ndarray, k: int = 10, ef_search: int = 50, num_threads: int = 0):
        th = num_threads if num_threads > 0 else (psutil.cpu_count(logical=True) or 4)
        faiss.omp_set_num_threads(th)
        q_data = np.ascontiguousarray(queries, dtype=np.float32)
        if self.metric == "cosine":
            norms = np.linalg.norm(q_data, axis=1, keepdims=True)
            q_data = (q_data / np.maximum(norms, 1e-12)).astype(np.float32)

        dists, ids = self.index.search(q_data, k)
        return ids.astype(np.int64), dists.astype(np.float32)
