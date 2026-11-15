# OrbitInfer

OrbitInfer is an inference runtime with a Rust scheduler, explicit cache
ownership and separately installable model runners.

At its core is the headless `Engine`: a fast Rust scheduler, KV cache, and
metrics server. You then write the forward pass as **model-specific kernels,
optimized for your hardware.**

```text
token IDs ──► Rust Engine ──► frozen BatchPlan ──► Python Runner ───► token IDs
                  ▲                                    │
                  └──────── ownership & commit ────────┘
```

OrbitInfer also has optional HTTP serving and Prometheus metrics.

> [!WARNING]
> This is not production-quality software. If you're looking for an actual
> inference engine, please use SGLang or vLLM. Read on if you want to hack on
> LLM inference systems, new models or hardware.

The complete source import includes CUDA model runners, MLX/Metal runners,
Torch reference models, serving tools, benchmarks and tests. The first planned
installed target is DeepSeek-V4.1-Flash on SM100. No complete GPU model profile
has been qualified for OrbitInfer. See the [plan](docs/completion-plan.md) and
[source import record](docs/source-import.md).

## Features

Start with the two-page [architecture overview](docs/architecture.md).

- [Rust batch scheduler](docs/kernel-boundary.md)
  - [Unified slab-based cache pool (KV, SWA, Recurrent)](docs/cache-pool.md)
    - [Pluggable prefix cache storage](docs/cache-storage.md)
    - [Hierarchical external KV design](docs/external-kv-design.md) (proposal)
  - [Speculative decoding (MTP, DSpark)](docs/model-forward-pass.md)
  - [Zero-overhead overlap scheduling](docs/scheduler.md)
  - [Engine metrics and Prometheus export](docs/metrics.md)
- [Parallelisms: TP, EP, CP, and DP Attention](docs/distributed.md)
- [Prefill and decode CUDA graphs](docs/cuda-graphs.md)
- [Native renderer and incremental parser](crates/orbitinfer-recipe/README.md)
  - [Recipe differential fuzzing](tests/recipe_fuzz/README.md)
- [NUMA pinning on worker startup](docs/numa.md)
- [Shared kernel cache](docs/kernel-cache.md)

## Getting started

Either fork the repository or import `orbitinfer` and `orbitinfer_models_cuda` as standalone
packages. Then use the following skills:

- [**add-orbitinfer-model**](skills/add-orbitinfer-model/SKILL.md) -
  Create an inference engine for a new model from scratch, using OrbitInfer.
- [**optimize-orbitinfer-performance**](skills/optimize-orbitinfer-performance/SKILL.md) -
  Iterate on your inference engine to improve performance with an "autoresearch"
  workflow, diagnose bottlenecks and validate improvements, drawing on past
  learnings.

Importing `orbitinfer` does not import PyTorch or initialize CUDA.

## Installation

Install the core library directly from GitHub with Python 3.11+ and Rust installed:

```bash
pip install .
```

Pip builds the Rust extension locally. The import remains `orbitinfer`.
Run this command from the local checkout; no package has been published.

Model runners are separate packages: [`orbitinfer-models-cuda`](models/cuda/pyproject.toml)
for optimized CUDA implementations,
[`orbitinfer-models-torch`](models/torch/pyproject.toml) for reference implementations,
and [`orbitinfer-models-mlx`](models/mlx/pyproject.toml) for Apple Silicon.
The repository's development workspace installs all three; the examples below assume
that workspace or a separately installed runner package.

## Development

OrbitInfer requires Python 3.11+ and Rust.

```bash
uv run scripts/rust_build_and_types.py
```

`uv run` syncs the project dependencies automatically. The build command then
syncs type-library sources and builds the Rust extension with generated stubs.

The Modal harnesses are launched from the repository root. Install the
development dependencies first, then authenticate with Modal once if this is a
new machine:

```bash
uv run modal setup
```

`modal setup` is interactive and only needs to be repeated when the local
Modal credentials change.

All models share one benchmark CLI and one parameterized serving app. The model
registry in [`deploy/models.py`](deploy/models.py) supplies checkpoint revisions,
hardware, parallelism, and model-specific runner construction.

```bash
# Populate the pinned checkpoint without allocating a GPU.
uv run scripts/bench.py --backend modal --model gpt-oss-20b --download-only

# Sweep prefill and decode, save traces, and plot the measured results.
uv run scripts/bench.py --backend modal --model gpt-oss-20b \
  --context-lengths 4096,8192 --batch-sizes 1,2,4,8 \
  --output bench/gpt-oss-20b.jsonl --trace-dir bench/traces --plot
```

For local GPUs, use `--backend local --checkpoint /path/to/checkpoint`; models
with a separate draft checkpoint also accept `--assistant-checkpoint`. The
local backend is the default. Modal submits the repository's source into the
shared CUDA image, so the client machine does not need CUDA installed.

Prefill charts compare complete-context latency with input tokens/s/GPU; decode
charts compare per-user output tokens/s with output tokens/s/GPU. Draft acceptance
uses normal verification; the plotter's `--accept-rate` projects other acceptance
rates. Model routing is the default; `--balanced-experts` enables synthetic
balanced routing. See
[benchmarking](docs/benchmarks.md) for the timing protocol and trace scope.

For an already running endpoint, `scripts/check_server.py URL --output REPORT`
checks serving correctness, prefix-cache reuse, context-size TTFT, and optional
image/reasoning/stress cases. It can also evaluate a small local MMLU CSV or
JSONL subset with `--eval-file` (defaults: `--eval-limit 100`,
`--eval-concurrency 32`); see
[server checks](docs/benchmarks.md#http-correctness-and-latency-checks).

## Minimal use

```python
from orbitinfer import Engine, SamplingParams
from orbitinfer_models_torch.lfm2 import Lfm2TorchRunner

runner = Lfm2TorchRunner(device="cpu")

with Engine(runner, context_length=32768) as engine:
    request = engine.submit(
        [1, 6, 24131],
        sampling_params=SamplingParams(
            max_output_tokens=8,
            stop_token_ids=[runner.eos_token_id],
            temperature=1.0,
            top_p=0.95,
        ),
    )
    while engine.has_work():
        for event in engine.step():
            print(event)
    generated_ids = request.take_result()
```

`SamplingParams` is the frozen request policy for generation limits, stop token
IDs, sampling, and logprob options. Use the recipe's `stop_token_ids`
for chat requests, or supply the model's stop tokens explicitly for raw token
requests. `Engine` has no default EOS token.

A custom synchronous runner only needs one method:

```python
import numpy as np

from orbitinfer import BatchPlan, ForwardOutput


class Runner:
    def forward(self, plan: BatchPlan) -> ForwardOutput:
        return ForwardOutput(
            token_ids=np.zeros(plan.num_sequences, dtype=np.uint32),
            token_indptr=np.arange(plan.num_sequences + 1, dtype=np.uint32),
        )
```

`ForwardOutput` carries flat token IDs and row offsets, with optional logprob
columns. The same result type covers ordinary and speculative decoding.

Managed history and recurrent state are cache groups declared through a frozen
`RunnerConfig`. The Engine binds their one physical arena during construction.
Pass `prefix_cache=True` to reuse aligned prefixes across requests.

## Local HTTP server

The built-in native server exposes one Engine through OpenAI Chat Completions and
Responses, Anthropic Messages, and a small demo chat page. Model-specific
prompt rendering and incremental response parsing live in the independent
`orbitinfer-recipe` Rust crate.

```bash
# Run one LFM2 prompt in the terminal.
uv run python scripts/lfm2_chat.py

# Serve LFM2 locally over HTTP.
uv run python scripts/lfm2_serve.py
```

Then open <http://127.0.0.1:8000>. The page reports prompt and generated token
counts, elapsed generation time, and output tokens per second. The same process
serves `/v1/chat/completions`, `/v1/responses`, `/v1/messages`, and `/v1/models`.
Both examples explicitly use the CPU Torch runner.

## Modal HTTP endpoints

[`deploy/modal_serve.py`](deploy/modal_serve.py) serves every registered model.
Select `ORBITINFER_MODEL` at launch; each selection has a separate app and endpoint,
with shared download, warmup, process management, and HTTP startup code.
Populate its pinned checkpoint before deploying:

```bash
ORBITINFER_MODEL=gpt-oss-20b uv run modal run deploy/modal_serve.py
ORBITINFER_MODEL=gpt-oss-20b uv run modal deploy deploy/modal_serve.py

ORBITINFER_MODEL=deepseek-v41 uv run modal run deploy/modal_serve.py
ORBITINFER_MODEL=deepseek-v41 uv run modal deploy deploy/modal_serve.py
```

Use `uv run scripts/bench.py --help` to see the current model keys accepted by
`ORBITINFER_MODEL` and the benchmark CLI. The shared
[`MODEL_SPECS` registry](deploy/models.py) defines each selection's checkpoint,
hardware, and serving limits.

## MLX runner (Apple Silicon)

Model runners are not tied to CUDA. `orbitinfer_models_mlx` runs
[dense and MoE Qwen3.5 models](models/mlx/orbitinfer_models_mlx/qwen3_5/README.md)
on Apple Silicon with MLX, on the same core engine. For a small local model:

```bash
QWEN_CHECKPOINT="$(uv run hf download mlx-community/Qwen3.5-0.8B-8bit \
  --revision 87e768fbfa03994095f3d14527c80c5ae70c5758 --quiet)"
uv run scripts/mlx_serve.py --model qwen3.5-0.8b-mlx --checkpoint "$QWEN_CHECKPOINT"
```

The download uses Hugging Face's normal cache and returns its snapshot path.

## Checks

```bash
cargo test --all-targets
uv run pytest -q
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run python scripts/cloc.py
```

GPU validation is model- and machine-specific; local tests use fakes or skip
unavailable accelerator libraries.

The code-size report uses Tokei to count production Rust/Python by component,
excluding comments, blank lines, tests, deployment tools, and generated stubs.
Use the current report instead of a checked-in line-count snapshot.

For resident prefill/decode sweeps, optional fixed speculative acceptance and
balanced routing, Perfetto traces, and throughput/latency charts, see
[benchmarking](docs/benchmarks.md).

## Acknowledgements

OrbitInfer was based on and inspired by systems from SGLang, vLLM, TokenSpeed, and DeepInfer.
