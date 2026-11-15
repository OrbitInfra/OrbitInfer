# Adapted for OrbitInfer; see NOTICE and docs/source-import.md.
from __future__ import annotations

import subprocess
import sys


class TestImport:
    def test_import_orbitinfer_does_not_import_accelerator_stacks(self) -> None:
        code = (
            "import sys; sys.path[:0] = ['models/cuda', 'models/torch']; "
            "import orbitinfer; import orbitinfer.backend; "
            "from orbitinfer import Parallelism; Parallelism(world_size=4, attn_dp=4); "
            "import orbitinfer.backend.cuda; import orbitinfer_models_cuda; "
            "import orbitinfer_models_torch; "
            "print(*(int(name in sys.modules) for name in "
            "('torch', 'flashinfer', 'flash_attn')))"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], check=True, capture_output=True, text=True
        )
        assert result.stdout.strip() == "0 0 0"
