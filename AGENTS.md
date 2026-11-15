# OrbitInfer agent guide

Read docs/architecture.md and docs/completion-plan.md before making changes.
This repository is an independent full-source adaptation of the snapshot in
third_party/upstream/import.json. Product names, modules and tools use OrbitInfer.
Preserve third-party copyright, licenses, notices and the source manifest.

## Owners

- One EngineCore owns scheduling, requests, logical pages and leases. Model
  runners own weights, arena backing, kernels, graphs and execution completion.
- A runner receives a frozen plan. Do not add a second scheduler, page allocator
  or forwarding coordinator. Storage providers extend the existing interface.
- OrbitKV owns external cache identity and policy. OrbiTE owns byte movement.
  The first OrbitKV adapter uses fixed-block GPU registrations and native receipts.
  Do not create fake success adapters or claim unsupported layouts.
- Cancel, timeout, logical completion and dropped handles are not physical
  retirement. Preserve actual backing until its last device/transport borrower
  retires. Uncertain cleanup retains ownership and stops admission.
- A rejected execution may retry only if it provably changed no persistent
  state. Never replay an uncertain WRITE or state-mutating model step.
- Bound requests, pages, storage operations and quarantined owners.

## Scope

The full source tree includes CUDA, MLX/Metal and Torch reference runners,
their tests, deployment tools, benchmarks and model-authoring skills. The first
deployment validation target remains DeepSeek-V4.1-Flash on SM100/Linux.
Imported source is not hardware qualification for any runner or platform.
Do not change OrbitKV or OrbiTE from this task. Do not publish packages or model
weights. Preserve original licenses and source hashes when importing changes.

## Checks

Use the pinned Rust toolchain and Python 3.12 development environment. Run
`uv sync --locked --no-default-groups --group test` and
`uv run --no-sync python scripts/check.py`. On Apple Silicon, `uv sync --locked`
also builds the MLX extension and installs all model workspace packages.
The script runs formatting, Clippy, Rust tests, Python tests, the installed CPU
example and source/package boundary controls. GPU skips are reported and never
counted as hardware evidence. Write installed gates before claiming GPU support.

Keep changes with their owner. Record source-import adaptations separately
from behavior changes. Tests should force lifecycle interleavings with events
or barriers. Never weaken an ownership assertion to make an import pass.
