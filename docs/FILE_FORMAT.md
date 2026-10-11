# HNSWDB File Format Specification (.hdb)

## 1. Overview
The `.hdb` binary file format provides deterministic, bit-identical index serialization with CRC32 integrity verification and atomic write guarantees.

All integers and floats are serialized in **Little-Endian** byte order.

---

## 2. Binary Layout

```
+-------------------------------------------------------------+
| Header (64 bytes)                                           |
|   Magic: "HNSWDB01" (8 bytes: 0x48 4E 53 57 44 42 30 31)     |
|   Format Version: uint32 (4 bytes) [Current = 2]            |
|   Metric: uint32 (4 bytes) [0=L2, 1=IP, 2=Cosine]           |
|   Dimension: uint32 (4 bytes)                               |
|   Current Elements: uint64 (8 bytes)                        |
|   Max Elements: uint64 (8 bytes)                            |
|   M: uint32 (4 bytes)                                       |
|   ef_construction: uint32 (4 bytes)                         |
|   ef_search: uint32 (4 bytes)                               |
|   entry_point: uint32 (4 bytes)                             |
|   max_level: int32 (4 bytes)                                |
|   seed: uint64 (8 bytes)                                    |
+-------------------------------------------------------------+
| Section 1: Labels Block                                     |
|   Length: uint64 (8 bytes = current_elements * 8)           |
|   int64[current_elements]                                   |
+-------------------------------------------------------------+
| Section 2: Levels Block                                     |
|   Length: uint64 (8 bytes = current_elements * 1)           |
|   uint8[current_elements]                                   |
+-------------------------------------------------------------+
| Section 3: Vectors Block                                    |
|   Length: uint64 (8 bytes = current_elements * dim * 4)     |
|   float32[current_elements * dim]                           |
+-------------------------------------------------------------+
| Section 4: Layer 0 Links Block                              |
|   Length: uint64 (8 bytes = current_elements * (1+2*M) * 4) |
|   uint32[current_elements * (1 + 2*M)]                      |
+-------------------------------------------------------------+
| Section 5: Upper Layer Links Block                          |
|   Length: uint64 (8 bytes)                                  |
|   For each node i with level > 0:                           |
|     node_idx: uint32                                        |
|     num_layers: uint32                                      |
|     For each level l in [0, num_layers - 1]:                |
|       count: uint32, links: uint32[count]                   |
+-------------------------------------------------------------+
| Section 6: Tombstones (Deleted) Block                       |
|   Length: uint64 (8 bytes = current_elements * 1)           |
|   uint8[current_elements] (0 = active, 1 = deleted)         |
+-------------------------------------------------------------+
| Section 7: Metadata Store Block                             |
|   Length: uint64 (8 bytes)                                  |
|   Field descriptors & length-prefixed records               |
+-------------------------------------------------------------+
| Checksum Trailer (4 bytes)                                  |
|   IEEE 802.3 CRC32 (uint32 little-endian)                   |
+-------------------------------------------------------------+
```

---

## 3. Data Types & Layout Details

### 3.1 Header (64 bytes)
- **Magic Bytes:** `0x48 0x4E 0x53 0x57 0x44 0x42 0x30 0x31` (`"HNSWDB01"`)
- **Version:** `0x00000002` (v2)
- Files with mismatched magic bytes or unsupported version numbers throw `CorruptFile` or `UnsupportedVersion` exceptions immediately on load.

### 3.2 Vectors & Topologies
- **Vectors Block:** Contiguous binary array of IEEE 754 32-bit floating point values. Size: `current_elements * dim * sizeof(float)`.
- **Layer 0 Block:** Contiguous array with stride `(1 + 2*M)` where each slice starts with `degree: uint32` followed by up to `2 * M` neighbor indices `idx_t: uint32`.
- **Upper Layers Block:** For each node $i$ with $\text{level} > 0$, serialized `(node_idx, num_layers, [(count, [links])])`.
- **Deleted Block:** `uint8_t` tombstone byte array per node (1 byte per internal ID).

---

## 4. Integrity & Atomic Write Protocol

### 4.1 CRC32 Verification
- The entire payload from byte 0 through the end of the metadata block is hashed with standard **IEEE 802.3 CRC32** polynomial (`0xEDB88320`).
- The 4-byte CRC32 is appended as the file trailer.
- On `load()`, the file is read, CRC32 is computed over all bytes preceding the trailer, and compared against the recorded checksum. Any bit corruption, byte flip, or truncation raises a `CorruptFile` exception.

### 4.2 Atomic Write Guarantee
To guarantee that crashes or power failures never leave a corrupt index file on disk:
1. The index is written to a temporary sibling file: `<target_path>.tmp.<pid>.<timestamp>`.
2. File buffers are flushed to disk using `fflush` / `FlushFileBuffers`.
3. The temporary file is atomically renamed to `<target_path>` using POSIX `rename()` or Windows `MoveFileEx(..., MOVEFILE_REPLACE_EXISTING)`.

---

## 5. Version History

- **v2 (Current):** Format version bumped to 2 following the CRC table correction. Saved files are validated against the standard IEEE 802.3 CRC32 polynomial (matching Python `zlib.crc32`).
- **v1:** Initial serialization format. v1 files were written with an incorrect CRC table. The loader rejects version-1 files with `UnsupportedVersion` and the message "re-save the index with the current version".
