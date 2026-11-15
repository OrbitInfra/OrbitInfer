# Adapted for OrbitInfer; see NOTICE and docs/source-import.md.
"""LFM2.5 model support for PyTorch."""

from orbitinfer_models_torch.lfm2.config import Lfm2Config
from orbitinfer_models_torch.lfm2.reference import (
    DEFAULT_MODEL,
    DEFAULT_REVISION,
    Lfm2TorchRunner,
)

__all__ = ["DEFAULT_MODEL", "DEFAULT_REVISION", "Lfm2Config", "Lfm2TorchRunner"]
