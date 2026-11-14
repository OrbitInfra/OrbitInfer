# Completion plan

## Current focus

The pinned Resin source import and local CPU validation are complete. The next
step is P0: verify final-owner retention and the native storage integration point.
No GPU execution or OrbitKV integration is accepted by copying source. See
[architecture](architecture.md) and [import evidence](source-import.md).

| Phase | Scope | State |
| --- | --- | --- |
| Foundation | Rust core, binding, shared NVIDIA helpers, DeepSeek source and provenance | Imported; local CPU checks pass; remote CI unrun |
| P0 | Final-reference/failed-completion lifetime audit, native storage injection and real backing ownership | Open; imported quarantine tests alone are insufficient |
| P1 | Full-checkpoint offline DeepSeek numerical comparison on SM100 | Unrun; hardware required |
| P2 | CUDA graphs, overlap, persistent-state faults and cancellation | Source imported; installed profile unqualified |
| P3 | OrbitKV adapter, DRAM first, then separate SSD and peer profiles through OrbiTE | Not implemented |
| P4 | Packaged serving and streaming lifecycle acceptance | HTTP/recipe sources imported; model-serving profile unqualified |
| P5 | Measured optimizations and separately audited Qwen/Kimi/GLM runners | Deferred |

## Installed validation

No GPU gate has run. Before a model profile is enabled, record its exact GPU,
driver, CUDA, Torch, kernel revisions, weight revision, mesh and numerical
tolerances. Verify full checkpoint execution, guards, cancellation, early drop,
failed synchronization and source/destination retention. A small tensor fixture
does not qualify a complete model.

DeepSeek baseline: fixed checkpoint from the [design](architecture-plan.md),
full prefill, no approximate SWA replay, no drafts, no overlap. Enable each
optimization separately. Compare natural routing and actual accepted tokens;
do not use forced draft acceptance as throughput evidence.
