# HNSWDB File Format Specification (.hnsw)

## 1. Overview
The `.hnsw` binary file format provides deterministic, bit-identical index serialization with CRC32 integrity verification and atomic write guarantees.

All integers and floats are serialized in **Little-Endian** byte order.

---

## 2. File Layout

```
+-------------------------------------------------------------+
| Header (32 bytes)                                           |
|   Magic: "HNSWDB\0\1" (8 bytes)                             |
|   Format Version: uint32 (4 bytes) [Current = 1]            |
|   Metric: uint32 (4 bytes) [0=L2, 1=IP, 2=Cosine]           |
|   Dimension: uint32 (4 bytes)                               |
|   Max Elements: uint64 (8 bytes)                            |
|   Current Elements: uint64 (8 bytes)                        |
|   Live Elements: uint64 (8 bytes)                           |
|   M: uint32 (4 bytes)                                       |
|   ef_construction: uint32 (4 bytes)                         |
|   ef_search: uint32 (4 bytes)                               |
|   max_level: int32 (4 bytes)                                |
|   entry_point: uint32 (4 bytes)                             |
|   seed: uint64 (8 bytes)                                    |
+-------------------------------------------------------------+
| Vectors Payload                                             |
|   float32[current_elements * dimension]                     |
+-------------------------------------------------------------+
| Node Properties                                             |
|   labels: int64[current_elements]                           |
|   levels: uint8[current_elements]                           |
|   deleted: uint8[current_elements]                          |
+-------------------------------------------------------------+
| Graph Topology                                              |
|   layer0_links: uint32[current_elements * (1 + 2*M)]        |
|   upper_layer_links: serialized jagged link lists           |
+-------------------------------------------------------------+
| Metadata Store                                              |
|   Field descriptors & JSON-compatible metadata records      |
+-------------------------------------------------------------+
| Checksum Trailer (4 bytes)                                  |
|   IEEE 802.3 CRC32 (uint32 little-endian)                   |
+-------------------------------------------------------------+
```

---

## 3. Data Types & Layout Details

### 3.1 Header (Magic & Version)
- **Magic Bytes:** `0x48 0x4E 0x53 0x57 0x44 0x42 0x00 0x01` (`"HNSWDB\0\1"`)
- **Version:** `0x00000001` (v1)
- Files with mismatched magic bytes or unsupported version numbers throw `CorruptFile` or `UnsupportedVersion` exceptions immediately on load.

### 3.2 Vectors & Topologies
- **Vectors Block:** Contiguous binary array of IEEE 754 32-bit floating point values. Size: `current_elements * dim * sizeof(float)`.
- **Layer 0 Block:** Contiguous array with stride `(1 + M0)` where each slice starts with `degree: uint32` followed by up to `2 * M` neighbor indices `idx_t: uint32`.
- **Upper Layers Block:** For each node $i$, `num_upper_levels: uint32`. For each level $l \in [1, \text{level}]$, `count: uint32` followed by `count` neighbors `idx_t: uint32`.

### 3.3 Metadata Block
- Field count `uint32`.
- For each record: `label_t: int64`, `field_name: string` (length-prefixed), `type_tag: uint8` (0=int64, 1=double, 2=string, 3=bool), followed by value payload.

---

## 4. Integrity & Atomic Write Protocol

### 4.1 CRC32 Verification
- The entire payload from byte 0 through the end of the metadata block is hashed with standard **IEEE 802.3 CRC32** polynomial (`0xEDB88320`).
- The 4-byte CRC32 is appended as the file trailer.
- On `load()`, the file is read into memory or streamed, CRC32 is calculated over all bytes preceding the trailer, and compared against the recorded checksum. Any bit corruption, byte flip, or truncation raises a `CorruptFile` exception.

### 4.2 Atomic Write Guarantee
To guarantee that crashes or power failures never leave a corrupt index file on disk:
1. The index is written to a temporary sibling file: `<target_path>.tmp.<pid>.<timestamp>`.
2. File buffers are flushed to disk using `fflush` / `FlushFileBuffers`.
3. The temporary file is atomically renamed to `<target_path>` using POSIX `rename()` or Windows `MoveFileEx(..., MOVEFILE_REPLACE_EXISTING)`.
