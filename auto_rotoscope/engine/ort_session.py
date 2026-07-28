"""onnxruntime session management and execution-provider selection.

CPU is always available. On Windows, `onnxruntime-directml` also exposes the
DirectML provider, which accelerates on any DirectX 12 GPU (NVIDIA/AMD/Intel)
without a CUDA install. GPU failures fall back to CPU cleanly.
"""

from __future__ import annotations

import os

# Imported lazily so that a missing wheel produces a clear, catchable error
# rather than crashing at add-on registration time.
try:
    import onnxruntime as ort
except Exception as exc:  # pragma: no cover - depends on bundled wheel
    ort = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class EngineNotAvailable(RuntimeError):
    """Raised when onnxruntime (bundled wheel) could not be imported."""


def is_available() -> bool:
    return ort is not None


def import_error() -> str:
    return "" if _IMPORT_ERROR is None else str(_IMPORT_ERROR)


def available_providers() -> list[str]:
    if ort is None:
        return []
    return list(ort.get_available_providers())


# Accelerated providers, in the order we prefer them. DirectML covers any
# DX12 GPU on Windows, CoreML the Apple GPU/Neural Engine on macOS. CUDA is
# listed for off-platform builds only — the CUDA wheel exceeds the 200 MB cap.
_GPU_PROVIDERS = ("DmlExecutionProvider", "CoreMLExecutionProvider", "CUDAExecutionProvider")


def has_gpu() -> bool:
    """True if a GPU-capable execution provider is present."""
    return bool(set(_GPU_PROVIDERS).intersection(available_providers()))


def _resolve_providers(mode: str) -> list[str]:
    """Return an ordered provider list for the requested mode.

    mode: 'AUTO' | 'CPU' | 'GPU'
    """
    avail = available_providers()
    cpu = ["CPUExecutionProvider"]

    if mode == "CPU":
        return cpu

    # Preferred GPU providers, in priority order, filtered by availability.
    gpu = [ep for ep in _GPU_PROVIDERS if ep in avail]

    if mode == "GPU":
        # Still append CPU as a safety net so a GPU init failure degrades
        # gracefully instead of raising.
        return gpu + cpu if gpu else cpu

    # AUTO
    return gpu + cpu


def make_session(model_path: str, provider_mode: str = "AUTO"):
    """Create an InferenceSession for `model_path` with the chosen providers.

    Returns (session, active_provider_name).
    """
    if ort is None:
        raise EngineNotAvailable(
            "onnxruntime is not available. The bundled wheel failed to import: "
            f"{import_error()}"
        )
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"ONNX model not found: {model_path}")

    providers = _resolve_providers(provider_mode)

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # Keep CPU threading reasonable; Blender may be doing other work.
    so.intra_op_num_threads = max(1, (os.cpu_count() or 4) // 2)

    session = ort.InferenceSession(model_path, sess_options=so, providers=providers)
    active = session.get_providers()[0] if session.get_providers() else "unknown"
    return session, active
