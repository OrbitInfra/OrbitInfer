# Completion plan

## Current focus

The complete pinned source tree is imported under OrbitInfer names: all 616
files, including CUDA, MLX/Metal and Torch model packages, tests, benchmarks,
deployment tools and authoring skills. Local validation is recorded in the
[import evidence](source-import.md). A complete source import does not establish
GPU model correctness, performance or production readiness.

Follow the [hierarchical external KV design](external-kv-design.md): one
Engine/page allocator, one selected provider and separate outcome/retirement
receipts. The proposal covers device residency, provider DRAM/SSD and peer
placement, with OrbitKV first, then independent Mooncake Store and LMCache
adapters. The first implementation makes load/store outcome and physical
retirement separate receipts through the existing Engine. No provider adapter
has been added.

Clean-environment validation includes the CPU test dependencies and model-source
paths inherited by subprocess controls. Current Clippy's nonzero-divisor checks
are addressed without changing scheduling arithmetic. Keep this CI baseline
passing before adding providers.

P0 remains open: the Python storage path now retains exact exported views on
failed retirement and final-reference drop, with forced-interleaving controls.
Next audit native/device execution backing and enforce representation identity
and borrowed-byte bounds. P3 begins with the CPU provider contract and OrbitKV
DRAM integration; SSD and peer profiles follow separately. GPU integration still
needs installed validation. The first deployment target remains DeepSeek on
SM100. Other imported runners are available for separate validation.

| Phase | Scope | State |
| --- | --- | --- |
| Foundation | Complete runtime, all model packages, tooling, tests and provenance | Source imported; validation in source-import.md; clean CPU gate repairs included |
| P0 | Final-reference/failed-completion lifetime audit, existing storage integration, real backing, identity and byte bounds | Partial: explicit Python storage receipts and backing-retention controls; native/device audit, identity and byte bounds remain open |
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
