"""Check complete source coverage, attribution and project namespaces."""

import ast
import hashlib
import json
import re
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise SystemExit(message)


def main():
    manifest = json.loads((ROOT / "third_party/upstream/import.json").read_text())
    require(
        manifest["revision"] == "a1989594d81921be3bffe43804eb0550b216ca64",
        "unexpected source revision",
    )
    require(
        not manifest["excluded_source_paths"],
        "the full snapshot must not exclude files",
    )
    paths = [entry["path"] for entry in manifest["files"]]
    require(
        len(paths) == len(set(paths)) == 616, "incomplete or duplicate source inventory"
    )
    for entry in manifest["files"]:
        require((ROOT / entry["path"]).is_file(), f"missing import: {entry['path']}")
        require(
            re.fullmatch("[0-9a-f]{64}", entry["source_sha256"]), "invalid source hash"
        )
    license_hash = hashlib.sha256((ROOT / "LICENSE").read_bytes()).digest()
    require(
        license_hash
        == hashlib.sha256(
            (ROOT / "third_party/upstream/LICENSE").read_bytes()
        ).digest(),
        "upstream license changed",
    )
    forbidden = {"resin", "resin_models_cuda", "resin_models_mlx", "resin_models_torch"}
    allowed = {
        "NOTICE",
        "docs/source-import.md",
        "scripts/check_source.py",
        "scripts/check.py",
    }
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
        if not path.is_file() or name.startswith("third_party/"):
            continue
        require("resin" not in name.lower(), f"old project path: {name}")
        if name not in allowed and "LICENSE" not in path.name and path.name != "NOTICE":
            try:
                text = path.read_text()
            except UnicodeDecodeError:
                continue
            require("resin" not in text.lower(), f"old project namespace in {name}")
        if path.suffix == ".py":
            for node in ast.walk(ast.parse(path.read_text(), filename=name)):
                modules = []
                if isinstance(node, ast.Import):
                    modules = [item.name for item in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    modules = [node.module]
                for module in modules:
                    require(
                        module.split(".")[0] not in forbidden,
                        f"upstream runtime import {module} in {name}",
                    )
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
            config["package"].get("publish") is False,
            f"publication is not disabled in {path}",
        )
    for backend in ("cuda", "mlx", "torch"):
        require(
            (ROOT / "models" / backend / f"orbitinfer_models_{backend}").is_dir(),
            f"missing {backend} runner package",
        )
    print(
        f"Source boundary passed: all {len(paths)} files; no exclusions or old runtime namespaces; licenses retained."
    )


if __name__ == "__main__":
    main()
