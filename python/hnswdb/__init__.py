"""HNSWDB - A high-performance approximate nearest neighbor vector database."""
from hnswdb._version import __version__

try:
    from hnswdb._core import (
        HNSWIndex,
        BuildOptions,
        FlatIndex,
        HnswdbError,
        DimensionMismatch,
        InvalidVector,
        CapacityExceeded,
        CorruptFile,
        UnsupportedVersion,
        InvalidArgument,
        active_kernel_name,
        set_force_scalar,
    )
except ImportError:
    pass  # C++ extension not built yet

__all__ = [
    "__version__",
    "HNSWIndex",
    "BuildOptions",
    "FlatIndex",
    "HnswdbError",
    "DimensionMismatch",
    "InvalidVector",
    "CapacityExceeded",
    "CorruptFile",
    "UnsupportedVersion",
    "InvalidArgument",
    "active_kernel_name",
    "set_force_scalar",
]

