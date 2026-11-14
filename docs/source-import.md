# Resin source import

The foundation is adapted from
[Resin at a1989594](https://github.com/ekzhang/resin/tree/a1989594d81921be3bffe43804eb0550b216ca64),
checked on 2026-11-14 and imported on 2026-11-15. The source archive SHA-256 is
`92464bd55a1b7c5f49291742b4dbaa365a52e78c6c4ca8000c638f9d4191733d`.
The [manifest](../third_party/resin/import.json) records every selected original
path and hash, the initial namespace-adapted hash, and every excluded path.

## Scope

342 upstream files are imported; 274 are excluded. Counts include tests,
fixtures, licenses and build metadata, not just production code.

| Imported owner | Files | Scope |
| --- | ---: | --- |
| `orbitinfer-core` | 20 | Rust scheduling, execution transactions, pages, leases and storage contracts |
| `orbitinfer-py` | 23 | PyO3 binding, request owners and frontend/session bridge |
| `orbitinfer-http` | 11 | Existing bounded HTTP frontend and its protocol controls |
| `orbitinfer-recipe` | 51 | Existing prompt/response formatting dependency and its fixtures |
| `orbitinfer` Python package | 64 | Shared execution/CUDA helpers, cache reference provider and DeepSeek configuration |
| NVIDIA model package and tests | 59 | DeepSeek-V4.1 runner, retained DeepInfer operators and model controls |
| Root Python tests and fixtures | 108 | Engine, completion, storage, transport, media and grammar controls |
| Root build/support files | 6 | Cargo metadata, test fixtures and license |

The import excludes MLX, CPU model runners, non-DeepSeek CUDA runners, upstream
deployment scripts, demo applications and benchmark CLIs. No model weights are
included. Generic recipe support is retained because it is a real binding
dependency; it does not advertise support for those models. Existing upstream
model reports remain labeled as upstream evidence, not OrbitInfer validation.

## Adaptations

- Rename Rust crates and Python modules to OrbitInfer; the native extension is
  `orbitinfer._native`. Rename paired protocol/HTML fields together. No Resin
  runtime package or compatibility alias is required.
- Keep the core wheel importable without Torch or CUDA. Model dependencies live
  in a separate NVIDIA package. CPU tensor tests install Torch through the test
  dependency group. Cargo and uv lockfiles fix the development environment.
- Retain the upstream Rust core, HTTP and recipe code together with the binding.
  No scheduler, allocator or coordinator is added.
- Simplify three boolean expressions equivalently for Clippy on Rust 1.96 and
  remove an obsolete PyO3 lint expectation. These change no runtime condition.
- Initialize Python explicitly in the Rust NumPy-borrow test, so it does not
  depend on another parallel test initializing the interpreter first. Preserve
  every assertion about the borrowed storage.
- Match the standard-library dataclass exception to Python 3.11/3.12 versus
  3.13+ while preserving the immutable-plan control. The local CPU test runner
  selects the loopback interface for child-process Gloo collectives.
- Remove model-specific test cases that instantiate excluded runners. Preserve
  their shared layout, topology and recipe controls. Exclude CLI/deployment tests
  whose owning scripts are not imported; the manifest lists the exact paths.
- Add a deterministic CPU runner example, an isolated installed-wheel check and
  a Linux/macOS CI definition. No fake OrbitKV adapter is provided.

The original Apache license and NOTICE are retained under `third_party/resin`.
DeepSeek MIT and DeepInfer Apache licenses stay next to their imported model
code. Packaging includes the applicable notices. Original source URLs remain
unchanged. Modified source files carry an adaptation notice. The upstream prompt
template keeps its terminal blank separator byte-for-byte; `.gitattributes`
exempts only that file's final blank line from whitespace checks.

## Validation and limits

Local host: macOS 26.6.2, arm64, Python 3.12.6, Rust 1.96.0. The source-boundary
check, formatting, strict Clippy and 310 Rust tests pass. Python reports 1,367
passed and 259 explicit skips: 236 require CUDA, two require Linux and 21 require
cached optional reference processors. The isolated installed wheel executes the
checked two-request example and imports without Torch, CUDA helpers or Resin.
The core sdist and separate model-source wheel build with their licenses.

The workflow is written but has not run on GitHub. Linux behavior, CUDA kernels,
complete model numerics and SM100 performance are not established by these local
checks. There is no installed GPU qualification or published package.

The next gate is the P0 lifetime/storage audit in the [plan](completion-plan.md):
prove final-reference retention of actual arena backing after failed device
completion, then implement the OrbitKV boundary on the same core. Imported
quarantine flags and CPU fixtures do not substitute for that proof.
