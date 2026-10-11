import os
import struct
import tempfile
import zlib
import numpy as np
import pytest

from hnswdb import HNSWIndex, CorruptFile, UnsupportedVersion


def test_crc_matches_zlib_trailer():
    """Save an index, read file bytes, and assert that zlib.crc32(data[:-4]) & 0xffffffff equals 4-byte trailer."""
    dim = 16
    n = 30
    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    ids = np.arange(n, dtype=np.int64)

    idx = HNSWIndex(dim=dim, metric="l2", max_elements=50, M=16, ef_construction=100, seed=42)
    idx.add(vecs, ids)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_crc.hdb")
        idx.save(path)

        with open(path, "rb") as f:
            data = f.read()

        assert len(data) >= 68  # 64-byte header + sections + 4-byte trailer
        trailer = struct.unpack("<I", data[-4:])[0]
        expected_crc = zlib.crc32(data[:-4]) & 0xFFFFFFFF
        assert trailer == expected_crc, f"CRC mismatch: trailer={trailer} vs zlib={expected_crc}"


def test_corrupted_byte_detected_by_crc():
    """Corrupting a byte in the saved index causes CRC verification failure."""
    dim = 16
    n = 20
    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    ids = np.arange(n, dtype=np.int64)

    idx = HNSWIndex(dim=dim, metric="l2", max_elements=50, M=16, ef_construction=100, seed=42)
    idx.add(vecs, ids)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_corrupt.hdb")
        idx.save(path)

        with open(path, "rb") as f:
            data = bytearray(f.read())

        # Corrupt a byte in the middle of payload
        data[35] ^= 0xFF
        corrupt_path = os.path.join(tmpdir, "corrupted.hdb")
        with open(corrupt_path, "wb") as f:
            f.write(data)

        with pytest.raises(CorruptFile):
            HNSWIndex.load(corrupt_path)


def test_truncated_file_detected():
    """Truncating a saved index causes load failure."""
    dim = 16
    n = 20
    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    ids = np.arange(n, dtype=np.int64)

    idx = HNSWIndex(dim=dim, metric="l2", max_elements=50, M=16, ef_construction=100, seed=42)
    idx.add(vecs, ids)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_trunc.hdb")
        idx.save(path)

        with open(path, "rb") as f:
            data = f.read()

        truncated_path = os.path.join(tmpdir, "truncated.hdb")
        with open(truncated_path, "wb") as f:
            f.write(data[: len(data) // 2])

        with pytest.raises(CorruptFile):
            HNSWIndex.load(truncated_path)


def test_version_1_file_rejected_with_message():
    """Version 1 files are rejected with UnsupportedVersion and instructions to re-save."""
    dim = 16
    n = 10
    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    ids = np.arange(n, dtype=np.int64)

    idx = HNSWIndex(dim=dim, metric="l2", max_elements=50, M=16, ef_construction=100, seed=42)
    idx.add(vecs, ids)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_v1.hdb")
        idx.save(path)

        with open(path, "rb") as f:
            data = bytearray(f.read())

        # Modify format version at offset 8 to 1
        struct.pack_into("<I", data, 8, 1)

        # Recompute CRC for the modified buffer
        new_crc = zlib.crc32(data[:-4]) & 0xFFFFFFFF
        struct.pack_into("<I", data, len(data) - 4, new_crc)

        v1_path = os.path.join(tmpdir, "v1.hdb")
        with open(v1_path, "wb") as f:
            f.write(data)

        with pytest.raises(UnsupportedVersion) as excinfo:
            HNSWIndex.load(v1_path)

        assert "re-save the index with the current version" in str(excinfo.value)
