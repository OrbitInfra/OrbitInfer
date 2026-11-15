# Adapted for OrbitInfer; see NOTICE and docs/source-import.md.
"""Measure cold Python import boundaries in fresh processes."""

from __future__ import annotations

import subprocess
import sys


def measure(statement: str) -> float:
    code = (
        "import time; start=time.perf_counter(); "
        f"{statement}; print((time.perf_counter()-start)*1000)"
    )
    output = subprocess.check_output([sys.executable, "-c", code], text=True)
    return float(output.strip())


print(f"import orbitinfer: {measure('import orbitinfer'):.2f} ms")
print(
    "import PyTorch backend module: "
    f"{measure('import orbitinfer_models_torch.lfm2.reference'):.2f} ms"
)
