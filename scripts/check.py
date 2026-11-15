"""Run the source-import gates. No GPU model or weights are loaded."""

import os
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args, env=None, cwd=ROOT):
    print("+", " ".join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), cwd=cwd, env=env, check=True)


def main():
    env = os.environ.copy()
    env["PYO3_PYTHON"] = sys.executable
    env["PYTHONPATH"] = sysconfig.get_path("platlib")
    run(sys.executable, "scripts/check_source.py")
    run("cargo", "fmt", "--all", "--check")
    run(
        "cargo",
        "clippy",
        "--locked",
        "--workspace",
        "--all-targets",
        "--",
        "-D",
        "warnings",
        env=env,
    )
    run("cargo", "test", "--locked", "--workspace", "--all-targets", env=env)
    run("cargo", "test", "--locked", "--workspace", "--doc", env=env)
    run(sys.executable, "-m", "ruff", "check", ".")
    # All distributed CPU fixtures use local child processes, not a cluster.
    cpu_env = os.environ.copy()
    cpu_env["GLOO_SOCKET_IFNAME"] = "lo0" if sys.platform == "darwin" else "lo"
    # The test group deliberately omits the GPU workspace packages. Subprocess
    # controls still import their Python source, just as pytest's pythonpath does.
    cpu_env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "models" / backend) for backend in ("cuda", "torch", "mlx")]
        + ([cpu_env["PYTHONPATH"]] if cpu_env.get("PYTHONPATH") else [])
    )
    run(sys.executable, "-m", "pytest", "-q", "-ra", env=cpu_env)
    with tempfile.TemporaryDirectory(prefix="orbitinfer-installed-") as temporary:
        out = Path(temporary)
        run(sys.executable, "-m", "maturin", "build", "--locked", "--out", out, env=env)
        wheels = list(out.glob("*.whl"))
        if len(wheels) != 1:
            raise RuntimeError(f"expected one wheel, got {wheels}")
        run(sys.executable, "-m", "venv", out / "venv")
        python = out / "venv/bin/python"
        run("uv", "pip", "install", "--python", python, wheels[0])
        run(python, "-I", ROOT / "examples/control_plane.py", cwd=out)
        run(
            python,
            "-I",
            "-c",
            "import sys, orbitinfer; "
            "imported = {'torch', 'resin', 'flashinfer', 'mlx'} & sys.modules.keys(); "
            "wrong = orbitinfer.Engine.__module__ != 'orbitinfer._native'; "
            "sys.exit(f'unexpected import: {imported}, binding={wrong}' if imported or wrong else 0)",
            cwd=out,
        )
        # CUDA packaging contains source only. The MLX package compiles its
        # native extension on macOS; it does not load any model checkpoint.
        for backend in ("cuda", "torch", "mlx"):
            run(
                "uv",
                "build",
                f"models/{backend}",
                "--wheel",
                "--out-dir",
                out / "models",
            )
        run(
            "uv",
            "pip",
            "install",
            "--python",
            python,
            "--no-deps",
            *sorted((out / "models").glob("*.whl")),
        )
        run(
            python,
            "-I",
            "-c",
            "import sys, orbitinfer_models_cuda, orbitinfer_models_mlx, orbitinfer_models_torch; "
            "loaded = {'torch', 'flashinfer', 'mlx'} & sys.modules.keys(); "
            "sys.exit(f'eager accelerator import: {loaded}' if loaded else 0)",
            cwd=out,
        )


if __name__ == "__main__":
    main()
