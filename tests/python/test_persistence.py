import os
import tempfile
import numpy as np
import pytest
import hnswdb
from hnswdb import HNSWIndex, FlatIndex, CorruptFile


def test_hnsw_save_load_roundtrip():
    dim = 32
    n = 200
    rng = np.random.default_rng(42)

    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    ids = np.arange(n, dtype=np.int64)

    # Metadata for each vector
    metadata = [{"tag": f"doc_{i}", "val": int(i * 10), "score": float(i) * 0.5, "active": (i % 2 == 0)} for i in range(n)]

    idx = HNSWIndex(dim=dim, metric="l2", max_elements=500, M=16, ef_construction=100, ef_search=50, seed=42)
    idx.add(vecs, ids, metadata=metadata)

    # Delete some items
    idx.delete([5, 15, 25])

    nq = 10
    queries = rng.standard_normal((nq, dim)).astype(np.float32)
    orig_ids, orig_dists = idx.search(queries, k=10)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_index.hdb")
        idx.save(path)

        loaded = HNSWIndex.load(path)
        assert len(loaded) == len(idx)
        assert loaded.dim() == idx.dim()

        # Search results must match
        load_ids, load_dists = loaded.search(queries, k=10)
        assert np.array_equal(orig_ids, load_ids)
        assert np.allclose(orig_dists, load_dists, atol=1e-5)

        # Vector contents must match
        for i in [0, 10, 50, 100]:
            assert np.allclose(idx.get_vector(i), loaded.get_vector(i), atol=1e-6)

        # Metadata must match
        assert loaded.get_metadata(10) == {"tag": "doc_10", "val": 100, "score": 5.0, "active": True}


def test_flat_save_load_roundtrip():
    dim = 16
    n = 50
    rng = np.random.default_rng(42)

    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    ids = np.arange(n, dtype=np.int64)

    flat = FlatIndex(dim=dim, metric="cosine")
    flat.add(vecs, ids)
    flat.delete([3, 7])

    nq = 5
    queries = rng.standard_normal((nq, dim)).astype(np.float32)
    orig_ids, orig_dists = flat.search(queries, k=5)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_flat.bin")
        flat.save(path)

        loaded = FlatIndex.load(path)
        assert len(loaded) == len(flat)

        load_ids, load_dists = loaded.search(queries, k=5)
        assert np.array_equal(orig_ids, load_ids)
        assert np.allclose(orig_dists, load_dists, atol=1e-5)


def test_corruption_detection():
    dim = 16
    n = 20
    vecs = np.random.randn(n, dim).astype(np.float32)
    ids = np.arange(n, dtype=np.int64)

    idx = HNSWIndex(dim=dim, metric="l2", max_elements=50)
    idx.add(vecs, ids)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "corrupt.hdb")
        idx.save(path)

        # Read file bytes
        with open(path, "rb") as f:
            data = bytearray(f.read())

        # 1. Flip magic header
        corrupt_magic = bytearray(data)
        corrupt_magic[0] = ord('X')
        bad_path1 = os.path.join(tmpdir, "bad_magic.hdb")
        with open(bad_path1, "wb") as f:
            f.write(corrupt_magic)

        with pytest.raises(CorruptFile):
            HNSWIndex.load(bad_path1)

        # 2. Corrupt a byte in the middle (payload corruption -> CRC32 mismatch)
        corrupt_payload = bytearray(data)
        corrupt_payload[100] ^= 0xFF
        bad_path2 = os.path.join(tmpdir, "bad_payload.hdb")
        with open(bad_path2, "wb") as f:
            f.write(corrupt_payload)

        with pytest.raises(CorruptFile):
            HNSWIndex.load(bad_path2)

        # 3. Truncated file
        truncated = data[:len(data) // 2]
        bad_path3 = os.path.join(tmpdir, "truncated.hdb")
        with open(bad_path3, "wb") as f:
            f.write(truncated)

        with pytest.raises(CorruptFile):
            HNSWIndex.load(bad_path3)


def test_binary_format_struct_offsets_and_crc():
    """Verify that saved .hdb files match the 64-byte header struct and CRC trailer."""
    import struct
    import zlib

    dim = 8
    n = 25
    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    ids = np.arange(n, dtype=np.int64)

    idx = HNSWIndex(dim=dim, metric="l2", max_elements=50, M=16, ef_construction=100, seed=42)
    idx.add(vecs, ids)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "format_test.hdb")
        idx.save(path)

        with open(path, "rb") as f:
            data = f.read()

        assert len(data) >= 68  # 64-byte header + sections + 4-byte CRC trailer

        # 1. Magic check (first 8 bytes)
        assert data[:8] == b"HNSWDB01"

        # 2. 64-byte header unpack:
        # 8s (magic), 3x I (version, metric, dim), 2x Q (count, max_elements),
        # 4x I (M, efc, efs, ep), 1x i (max_lvl), 1x Q (seed)
        # Total: 8 + 12 + 16 + 16 + 4 + 8 = 64 bytes
        header_fmt = "<8sIIIQQIIIIiQ"
        assert struct.calcsize(header_fmt) == 64

        (magic, version, metric, d, count, max_el,
         M, efc, efs, ep, max_lvl, seed) = struct.unpack(header_fmt, data[:64])

        assert magic == b"HNSWDB01"
        assert version == 2
        assert metric == 0  # L2
        assert d == dim
        assert count == n
        assert max_el == 50
        assert M == 16
        assert efc == 100
        assert seed == 42

        # 3. CRC trailer check (last 4 bytes, IEEE 802.3 CRC32)
        stored_crc = struct.unpack("<I", data[-4:])[0]
        payload = data[:-4]
        computed_crc = zlib.crc32(payload) & 0xFFFFFFFF
        assert stored_crc == computed_crc

