# OrbitInfer agent guide

Read docs/architecture.md and docs/completion-plan.md before making changes.
This repository is an independent, NVIDIA-only source adaptation of Resin at
the revision recorded in third_party/resin/import.json. Preserve attribution.

## Owners

- One EngineCore owns scheduling, requests, logical pages and leases. Model
  runners own weights, arena backing, kernels, graphs and execution completion.
- A runner receives a frozen plan. Do not add a second scheduler, page allocator
  or forwarding coordinator. Storage providers extend the existing interface.
- OrbitKV owns external cache identity and policy. OrbiTE owns byte movement.
  Their integration is not yet implemented; do not create fake success adapters.
- Cancel, timeout, logical completion and dropped handles are not physical
  retirement. Preserve actual backing until its last device/transport borrower
  retires. Uncertain cleanup retains ownership and stops admission.
- A rejected execution may retry only if it provably changed no persistent
  state. Never replay an uncertain WRITE or state-mutating model step.
- Bound requests, pages, storage operations and quarantined owners.

## Scope

First model: DeepSeek-V4.1-Flash; first installed target: SM100 on Linux.
CPU fixtures are permitted; production CPU, Ascend, MLX/Metal and ROCm runners
are excluded. Other CUDA runners are future, separately verified additions.
Do not change OrbitKV or OrbiTE from this task. Do not publish packages or model
weights. Preserve original licenses and source hashes when importing changes.

## Checks

Use the pinned Rust toolchain and Python 3.12 development environment. Run
`uv sync --locked --group test` and `uv run --no-sync python scripts/check.py`.
The script runs formatting, Clippy, Rust tests, Python tests, the installed CPU
example and source/package boundary controls. GPU skips are reported and never
counted as hardware evidence. Write installed gates before claiming GPU support.

Keep changes with their owner. Record source-import adaptations separately
from behavior changes. Tests should force lifecycle interleavings with events
or barriers. Never weaken an ownership assertion to make an import pass.
