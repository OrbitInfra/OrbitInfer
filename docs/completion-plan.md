# Completion plan

## Current focus

OrbitInfer maintains its runtime, CUDA, MLX/Metal and Torch model packages,
tests, benchmarks and deployment tools independently. Agent skills live in
`.agents/skills/`; external updates are reviewed manually as needed. Attribution
and the initial validation record remain in [source notes](source-import.md).
Source availability does not establish GPU model correctness, performance or
production readiness.

Follow the [hierarchical external KV design](external-kv-design.md): one
Engine/page allocator, one selected provider and separate outcome/retirement
receipts. The proposal covers device residency, provider DRAM/SSD and peer
placement, with OrbitKV first, then independent Mooncake Store and LMCache
adapters. The first implementation makes load/store outcome and physical
retirement separate receipts through the existing Engine. A separate OrbitKV adapter maps the existing client to fixed, full-history GPU
blocks in same-host DRAM; its installed profile is unqualified.

Clean-environment validation includes the CPU test dependencies and model-source
paths inherited by subprocess controls. Current Clippy's nonzero-divisor checks
are addressed without changing scheduling arithmetic. Keep this CI baseline
passing before adding providers.

P0 is implemented at the Rust/Python owner boundary: final drops retain actual
backing, uncertain completions and page charges; storage keys bind representation
identity and borrowed bytes are bounded until retirement. CPU controls include
synchronous failure, asynchronous inspection/harvest failure and a successful
release control. Installed CUDA execution remains unqualified. P3 now has CPU contract controls for the fixed-block OrbitKV DRAM adapter; SSD and peer profiles follow separately. GPU integration still
needs installed validation. The first deployment target remains DeepSeek on
SM100. Other imported runners are available for separate validation.

| Phase | Scope | State |
| --- | --- | --- |
| Foundation | Runtime, model packages, tooling, tests and attribution | Maintained independently; installed-wheel CPU check retained; no source-mirror manifest |
| P0 | Final-reference/failed-completion lifetime audit, existing storage integration, real backing, identity and byte bounds | Implemented at source level; real backing, identity, byte and metadata bounds verified on CPU; installed CUDA gate unrun |
| P1 | Full-checkpoint offline DeepSeek numerical comparison on SM100 | Unrun; hardware required |
| P2 | CUDA graphs, overlap, persistent-state faults and cancellation | Source imported; installed profile unqualified |
| P3 | OrbitKV DRAM, then SSD/peer through OrbiTE; separate Mooncake Store and LMCache adapters | Local DRAM adapter implemented for one fixed full-history GPU group; CPU client/CUDA shim controls; installed SM100 gate unrun; SSD/peer/window/state mappings deferred |
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

### External-cache installed gate

`uv run --no-sync python tests/hardware/cache_storage_cuda.py --device cuda:0`
checks GPU views, guards, prefix restore and final-owner quarantine on SM100.
Written but not run. It does not qualify full-model execution or OrbitKV.

OrbitKV local DRAM: `uv run --no-sync python tests/hardware/orbitkv_dram.py
--bootstrap-socket /run/orbitkv/orbitkv.sock --metrics-url http://127.0.0.1:9090/metrics
--device cuda:0 --output /tmp/orbitinfer-orbitkv-dram.json`.
Requires a DRAM-only Manager; written but not run. See [layout and route limits](orbitkv-dram.md).
