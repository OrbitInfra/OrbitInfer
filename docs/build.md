# Build and CPU checks

The source import uses Rust 1.96 and Python 3.12. The core has no Python or GPU
runtime dependency; the binding is abi3 for Python 3.11 and newer. That ABI choice
is inherited, not a claim that all Python/GPU combinations have been tested.

```bash
uv sync --locked --group test
uv run --no-sync python scripts/check.py
```

The test group installs CPU-capable Torch for tensor/layout controls. It does
not resolve or install the separate CUDA model package. Importing `orbitinfer` must not
import Torch or load an accelerator library. `examples/control_plane.py` runs
through the compiled binding with a deterministic test runner and checked tokens.

Individual commands:

```bash
cargo fmt --all --check
PYO3_PYTHON="$PWD/.venv/bin/python" cargo clippy --locked --workspace --all-targets -- -D warnings
PYO3_PYTHON="$PWD/.venv/bin/python" cargo test --locked --workspace --all-targets
uv run --no-sync pytest -q
uv run --no-sync maturin build --locked --out dist
```

Rust tests that embed Python also need the environment's site-packages on
`PYTHONPATH`; `scripts/check.py` supplies it. The test group needs no GPU, even
when the selected platform's Torch wheel includes CUDA runtime dependencies.

## NVIDIA dependencies

The separate `models/cuda` package retains the pinned DeepSeek kernel stack.
Install it only on the Linux SM100 validation host, against the selected CUDA
toolchain. Optional DeepGEMM/FlashMLA/DeepSelect sources keep their upstream
revisions. They are not fetched or built by the core wheel build.

This initial import does not provide an installed GPU qualification. Model
weights are not included. No package is published by the development scripts.
