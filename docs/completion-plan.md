# Completion plan

## Current focus

The complete pinned source tree is imported under OrbitInfer names: all 616
files, including CUDA, MLX/Metal and Torch model packages, tests, benchmarks,
deployment tools and authoring skills. Local validation is recorded in the
[import evidence](source-import.md). A complete source import does not establish
GPU model correctness, performance or production readiness.

Review the [hierarchical external KV design](external-kv-design.md) first: one
Engine/page allocator, one selected provider and separate outcome/retirement
receipts. The proposal covers device residency, provider DRAM/SSD and peer
placement, with OrbitKV first, then independent Mooncake Store and LMCache
adapters. It adds no adapter implementation.

Before provider implementation, make the clean-environment CI baseline pass.
The first import runs exposed newer-Clippy diagnostics and missing test/runtime
dependencies; local passes do not substitute for that gate.

The next implementation step remains P0: verify final-owner retention through
the existing native/Python storage path, then enforce representation identity
and borrowed-byte bounds. P3 begins with the CPU provider contract and OrbitKV
DRAM integration; SSD and peer profiles follow separately. GPU integration still
needs installed validation. The first deployment target remains DeepSeek on
SM100. Other imported runners are available for separate validation.

| Phase | Scope | State |
| --- | --- | --- |
| Foundation | Complete runtime, all model packages, tooling, tests and provenance | Source imported; local validation in source-import.md; CI environment/toolchain fixes pending |
| P0 | Final-reference/failed-completion lifetime audit, existing storage integration, real backing, identity and byte bounds | Open; imported quarantine tests alone are insufficient |
| P1 | Full-checkpoint offline DeepSeek numerical comparison on SM100 | Unrun; hardware required |
| P2 | CUDA graphs, overlap, persistent-state faults and cancellation | Source imported; installed profile unqualified |
| P3 | OrbitKV DRAM, then SSD/peer through OrbiTE; separate Mooncake Store and LMCache adapters | Design proposed; not implemented; sequence in external-kv-design.md |
| P4 | Packaged serving and streaming lifecycle acceptance | Sources imported; model-serving profile unqualified |
| P5 | Other CUDA, Torch and MLX runners; measured optimizations | Sources imported; complete-model validation deferred |

Existing TP/attention-DP/CP/EP/expert-TP paths remain subject to runner capability
checks and installed validation. PP and independent E/P/D deployment are not
implemented and are outside the external-cache step. Do not infer them from
distributed rank layout or from restoring a stored prefix.

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
