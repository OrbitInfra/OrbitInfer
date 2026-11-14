# OrbitInfer

An NVIDIA inference runtime with Rust scheduling and explicit resource ownership.

OrbitInfer starts from [Resin](https://github.com/ekzhang/resin) at
`a1989594d81921be3bffe43804eb0550b216ca64`. It retains one Rust engine and page
allocator, replaceable model runners, and an external prefix-storage interface.
The first imported model is DeepSeek-V4.1-Flash on SM100. The CUDA execution
sources are imported; no OrbitInfer GPU profile has been qualified.

## Architecture

- `crates/orbitinfer-core`: request lifecycle, scheduling, plans, page leases.
- `crates/orbitinfer-py`: `orbitinfer._native`, bound directly to that core.
- `crates/orbitinfer-http` and `orbitinfer-recipe`: retained serving and formatting
  dependencies of the binding. They do not select or load model weights.
- `orbitinfer/backend`: execution plans, completion, distributed and CUDA helpers.
- `models/cuda/orbitinfer_models_cuda/deepseek_v41`: the NVIDIA model runner.
- `CacheStorage` / native `PrefixStorage`: the external-cache extension point.
  OrbitKV integration is planned, not implemented by this import.

OrbitInfer owns inference and HBM page leases. OrbitKV will own external cache
identity, replicas and policy. OrbiTE will own actual payload copies and retirement.
See the [architecture](docs/architecture.md), [source audit and plan](docs/architecture-plan.md),
[build guide](docs/build.md), and [import record](docs/source-import.md).

## CPU development

The control plane and its fixtures run without a GPU. This example is a
deterministic test runner, not a language model or CPU inference backend.

```bash
uv sync --locked --group test
uv run --no-sync python examples/control_plane.py
uv run --no-sync python scripts/check.py
```

Production model execution targets NVIDIA on Linux. MLX, Ascend and CPU model
runners are not imported. The core package does not import Torch or initialize
CUDA at import time. NVIDIA model dependencies live in the separate model package.

## Status

Initial source import; unpublished. CPU checks cover the retained owners, not
GPU execution. Full-checkpoint numerics, failure-time final-owner retention,
OrbitKV integration and installed SM100 validation remain explicit gates in the
[plan](docs/completion-plan.md).

## License

Apache-2.0 with retained third-party notices. See [LICENSE](LICENSE), [NOTICE](NOTICE)
and the original [Resin notices](third_party/resin/NOTICE). No model weights are included.
