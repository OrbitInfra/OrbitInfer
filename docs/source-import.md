# Source import

OrbitInfer adapts the complete source snapshot of
[Resin at a1989594](https://github.com/ekzhang/resin/tree/a1989594d81921be3bffe43804eb0550b216ca64),
checked on 2026-11-14 and imported on 2026-11-15. The source archive SHA-256 is
`92464bd55a1b7c5f49291742b4dbaa365a52e78c6c4ca8000c638f9d4191733d`.
The [manifest](../third_party/upstream/import.json) maps all 616 upstream files
to their local paths and records original and namespace-transformed hashes.
There are no excluded source paths. The previous subset selected 342 files;
the complete import restores the other 274 files and previously trimmed tests.

## Scope and naming

All four Rust crates, the Python binding, all shared runtime helpers, CUDA,
MLX/Metal and Torch model packages, tests, fixtures, docs, deployment scripts,
benchmarks, workflows and authoring skills are present. No model weights or
upstream Git history are included. The source snapshot, rather than a moving
branch, defines this import.

Product names, Python imports, Rust crates, environment names, metrics,
protocol fields, kernel symbols, cache paths and tool names use OrbitInfer.
The native Python module is `orbitinfer._native`; there is no upstream runtime
package or compatibility alias. Original project names remain only in legal
notices, provenance records and checks which reject old namespaces.

Original licenses, copyright and NOTICE text are retained. Source and model
packages include their applicable notices. Upstream benchmark numbers in
imported guides are historical reference measurements, not OrbitInfer results.
See the [historical audit](../third_party/upstream/source-audit.md) for the
original NVIDIA-only evaluation; the full-source scope supersedes its exclusions.

## Adaptations

- Rename project identifiers consistently in Python, Rust, C++, CUDA, Metal,
  build metadata, templates and fixtures; retain algorithms and ownership.
- Keep the existing equivalent Clippy boolean simplifications, Rust formatting,
  and removal of an obsolete PyO3 lint expectation for the pinned Rust toolchain.
- Keep explicit Python initialization in the Rust NumPy-borrow control, and the
  CPython-version-specific dataclass exception check. No assertion is removed.
- Restore the complete upstream Python workspace and all model/test entrypoints.
  Retain a CPU-only test dependency group for machines without native GPU SDKs.
- Use pytest's importlib mode so CUDA and MLX tests with the same basename can
  run together. The MLX ordering control explicitly skips when MLX is absent,
  matching the other MLX controls; the Apple workflow requires Metal and cannot
  silently accept missing MLX coverage.
- Keep local development versions and `publish = false` for Rust crates. There
  is no remote repository or package publication in this step.
- Preserve the deterministic CPU example and isolated installed-wheel check.
  CPU distributed controls select the local loopback interface.
- Replace the upstream README's performance promotion with the current
  qualification limits. Existing owner documentation remains in place.

## Validation and limits

Local host: macOS arm64, Python 3.12.6, Rust 1.96.0. All 616 original file
hashes match the pinned archive and every mapped local file is present.

| Check | Result |
| --- | --- |
| Namespace, complete import and original license checks | Pass |
| Rust formatting, strict Clippy, workspace tests and doctests | Pass; 310 tests |
| Python Ruff, formatting and `ty` with locked library type sources | Pass |
| Complete Python suite | 2,405 passed; 551 explicit skips |
| MLX native build and included small Metal controls | Pass; 59 MLX tests within the full suite |
| Core, CUDA, Torch and MLX wheels; core sdist | Built with original notices |
| Isolated installed core CPU example | Pass; two requests and checked token results |
| Isolated installed package imports | Pass; no eager accelerator or upstream runtime import |

Python passes comprise 1,614 shared controls, 713 CUDA-package CPU controls,
59 MLX controls and 19 Torch controls. Skips comprise 524 CUDA/hardware checks,
two Linux checks and 25 checks requiring uncached reference processors or
checkpoints. No skip is counted as hardware evidence. The missing local Metal
compiler component was installed to compile and execute the MLX controls.

No complete GPU model has been run, no deployment tool has been executed
against a cloud account, and no package has been published. Linux and remote CI
are unrun. The [completion plan](completion-plan.md) separates this source
import from installed model validation and future OrbitKV/OrbiTE integration.
