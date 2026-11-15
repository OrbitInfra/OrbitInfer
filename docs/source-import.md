# Source and license notes

OrbitInfer is maintained independently. The original source revision and
third-party attribution are recorded in [NOTICE](../NOTICE); source and model
packages retain their applicable licenses and notices. The original file
inventory remains available in Git history and is not a maintenance contract.

Product namespaces and package imports use OrbitInfer. Renamed and adapted
files retain their change notices. The core, CUDA, MLX/Metal and Torch packages
remain separate; no external inference runtime is required at import time.

External changes are inspected manually when relevant. Record the exact
revision and tests in the change review; there is no automatic synchronization.
Review ownership, cache identity and resource bounds before adopting a change.
Existing GPU benchmark tables from the initial source are historical reference
measurements, not OrbitInfer qualification.

## Initial validation record (2026-11-15)

At the initial import: macOS arm64, Python 3.12.6, Rust 1.96.0. The complete
source snapshot and its original file hashes were verified at that time. This
historical evidence does not require retaining a source mirror or manifest.

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

At that checkpoint, no complete GPU model or cloud deployment had run, no
package had been published, and Linux/remote CI were unrun. Current status is
in the [completion plan](completion-plan.md); this table is historical evidence.
