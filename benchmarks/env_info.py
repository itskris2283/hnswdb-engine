import os
import platform
import subprocess
import sys
import importlib.metadata
import psutil

try:
    import numpy as np
except ImportError:
    np = None

try:
    import faiss
except ImportError:
    faiss = None

try:
    import hnswlib
except ImportError:
    hnswlib = None

try:
    import hnswdb
    from hnswdb import _core
except ImportError:
    hnswdb = None
    _core = None


def get_git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def get_cpu_model() -> str:
    model = platform.processor()
    if platform.system() == "Windows":
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            model, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            return model.strip()
        except Exception:
            pass
    elif platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":", 1)[1].strip()
        except Exception:
            pass
    return model or platform.machine()


def get_version(pkg_name: str, module=None) -> str:
    if module is not None and hasattr(module, "__version__"):
        return str(module.__version__)
    try:
        return importlib.metadata.version(pkg_name)
    except Exception:
        return "unknown"


def get_env_info() -> dict:
    freq = psutil.cpu_freq()
    base_clock_mhz = freq.current if freq else None

    # Memory in GiB
    ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)

    # Process affinity and priority
    proc = psutil.Process()
    try:
        cpu_affinity = proc.cpu_affinity()
    except Exception:
        cpu_affinity = list(range(psutil.cpu_count(logical=True) or 1))

    try:
        niceness = proc.nice()
    except Exception:
        niceness = None

    # Active kernel
    simd_kernel = "unknown"
    if _core is not None and hasattr(_core, "active_kernel_name"):
        try:
            simd_kernel = _core.active_kernel_name()
        except Exception:
            pass

    info = {
        "cpu_model": get_cpu_model(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "cpu_clock_mhz": base_clock_mhz,
        "ram_gb": ram_gb,
        "os": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "compiler": "MSVC 2022 (cl.exe 19.40)" if platform.system() == "Windows" else "GCC/Clang",
        "cmake_build_type": "Release",
        "cmake_flags": "/O2 /MD /arch:AVX2 (MSVC)" if platform.system() == "Windows" else "-O3 -march=native",
        "git_commit": get_git_commit(),
        "active_simd_kernel": simd_kernel,
        "process_affinity": cpu_affinity,
        "process_priority": str(niceness),
        "versions": {
            "hnswdb": get_version("hnswdb", hnswdb),
            "numpy": get_version("numpy", np),
            "faiss": get_version("faiss-cpu", faiss),
            "hnswlib": get_version("hnswlib", hnswlib),
        }
    }
    return info


if __name__ == "__main__":
    import json
    info = get_env_info()
    print(json.dumps(info, indent=2))
