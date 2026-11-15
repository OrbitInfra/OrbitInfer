# Completion plan

## Current focus

The complete pinned source tree is imported under OrbitInfer names: all 616
files, including CUDA, MLX/Metal and Torch model packages, tests, benchmarks,
deployment tools and authoring skills. Local validation is recorded in the
[import evidence](source-import.md). A complete source import does not establish
GPU model correctness, performance or production readiness.

The next implementation step remains P0: verify final-owner retention and the
native storage integration point. The first deployment target remains DeepSeek
on SM100. Other imported runners are available for separate validation.

| Phase | Scope | State |
| --- | --- | --- |
| Foundation | Complete runtime, all model packages, tooling, tests and provenance | Source imported; validation in source-import.md |
| P0 | Final-reference/failed-completion lifetime audit, native storage injection and real backing ownership | Open; imported quarantine tests alone are insufficient |
| P1 | Full-checkpoint offline DeepSeek numerical comparison on SM100 | Unrun; hardware required |
| P2 | CUDA graphs, overlap, persistent-state faults and cancellation | Source imported; installed profile unqualified |
| P3 | OrbitKV adapter, DRAM first, then separate SSD and peer profiles through OrbiTE | Not implemented |
| P4 | Packaged serving and streaming lifecycle acceptance | Sources imported; model-serving profile unqualified |
| P5 | Other CUDA, Torch and MLX runners; measured optimizations | Sources imported; complete-model validation deferred |

## Installed validation

Before a model profile is enabled, record its exact device, driver, runtime,
kernel revisions, weight revision, mesh and numerical tolerances. Verify full
checkpoint execution, guards, cancellation, early drop, failed synchronization
and source/destination retention. Small tensor fixtures do not qualify a model.

DeepSeek baseline: full prefill (`swa_replay_mode="off"`), no speculative drafts
and no overlap. Enable each optimization separately. Compare natural routing
and actual accepted tokens; forced draft acceptance is not throughput evidence.

The imported deployment scripts and skills are tools, not authorization to run
cloud jobs, download complete checkpoints or publish packages. OrbitKV and
OrbiTE integrations remain separate work.
