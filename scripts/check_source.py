"""Check first-party package boundaries and retained license notices."""

import ast
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise SystemExit(message)


def main():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    require(project["name"] == "orbitinfer", "unexpected core package name")
    for pattern in project["license-files"]:
        require(any(ROOT.glob(pattern)), f"missing packaged notice: {pattern}")
    license_text = (ROOT / "LICENSE").read_bytes()
    notice = (ROOT / "NOTICE").read_bytes()
    require(
        bool(license_text.strip()) and bool(notice.strip()), "empty license or notice"
    )
    names = (
        subprocess.check_output(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            cwd=ROOT,
        )
        .decode()
        .split("\0")
    )
    for name in sorted(set(names) - {""}):
        path = ROOT / name
        if not path.is_file():
            continue
        if path.suffix == ".py":
            for node in ast.walk(ast.parse(path.read_text(), filename=name)):
                modules = []
                if isinstance(node, ast.Import):
                    modules = [item.name for item in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    modules = [node.module]
                for module in modules:
                    for backend in ("cuda", "mlx", "torch"):
                        prefix = f"orbitinfer_models_{backend}"
                        if module == prefix or module.startswith(prefix + "."):
                            base = ROOT / "models" / backend / Path(*module.split("."))
                            require(
                                base.is_dir()
                                or base.with_suffix(".py").is_file()
                                or base.with_suffix(".pyi").is_file(),
                                f"missing model module {module} in {name}",
                            )
    for path in (ROOT / "crates").glob("*/Cargo.toml"):
        config = tomllib.loads(path.read_text())
        require(
            config["package"]["name"].startswith("orbitinfer-"),
            f"unexpected crate name in {path}",
        )
        require(
            config["package"].get("publish") is False,
            f"publication is not disabled in {path}",
        )
    for backend in ("cuda", "mlx", "torch"):
        package = ROOT / "models" / backend
        config = tomllib.loads((package / "pyproject.toml").read_text())
        require(
            config["project"]["name"] == f"orbitinfer-models-{backend}",
            f"unexpected {backend} package name",
        )
        require((package / "LICENSE").read_bytes() == license_text, "license changed")
        require((package / "NOTICE").read_bytes() == notice, "package notice differs")
        require(
            (ROOT / "models" / backend / f"orbitinfer_models_{backend}").is_dir(),
            f"missing {backend} runner package",
        )
    print("Source boundary passed: package names, model modules and license notices.")


if __name__ == "__main__":
    main()
