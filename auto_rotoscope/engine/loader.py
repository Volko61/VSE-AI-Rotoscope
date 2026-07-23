"""Lazy, cached loading of the SAM 2.1 encoder/decoder sessions."""

from __future__ import annotations

import os

from . import ort_session
from .sam2 import SAM2Image

_MODELS_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

# Cache keyed by (model_dir, provider_mode) so switching device rebuilds it.
_cache: dict[tuple[str, str], SAM2Image] = {}
_active_provider: str = ""


def active_provider() -> str:
    return _active_provider


def clear_cache():
    global _cache
    _cache.clear()


def load(model_dir: str, provider_mode: str) -> SAM2Image:
    """Return a ready SAM2Image for the given model directory and device mode."""
    global _active_provider

    key = (model_dir, provider_mode)
    if key in _cache:
        return _cache[key]

    root = os.path.join(_MODELS_ROOT, model_dir)
    encoder = os.path.join(root, "encoder.onnx")
    decoder = os.path.join(root, "decoder.onnx")

    for path in (encoder, decoder):
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Missing model file: {path}. Run scripts/fetch_assets.py before building."
            )

    enc_sess, prov = ort_session.make_session(encoder, provider_mode)
    dec_sess, _ = ort_session.make_session(decoder, provider_mode)
    _active_provider = prov

    sam = SAM2Image(enc_sess, dec_sess)
    _cache[key] = sam
    return sam
