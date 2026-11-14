"""Check import provenance, local module references and excluded runtimes."""

import ast
import hashlib
import json
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise SystemExit(message)


def main():
    manifest = json.loads((ROOT / "third_party/resin/import.json").read_text())
    require(manifest["revision"] == "a1989594d81921be3bffe43804eb0550b216ca64", "unexpected source revision")
    paths = [entry["path"] for entry in manifest["files"]]
    require(len(paths) == len(set(paths)), "duplicate imported path")
    for entry in manifest["files"]:
        require((ROOT / entry["path"]).is_file(), f"missing import: {entry['path']}")
        require(re.fullmatch("[0-9a-f]{64}", entry["source_sha256"]), "invalid source hash")
    license_hash = hashlib.sha256((ROOT / "LICENSE").read_bytes()).digest()
    require(license_hash == hashlib.sha256((ROOT / "third_party/resin/LICENSE").read_bytes()).digest(), "upstream license changed")
    forbidden = {"resin", "resin_models_cuda", "resin_models_mlx", "resin_models_torch", "mlx", "torch_npu"}
    for folder in ("orbitinfer", "models/cuda/orbitinfer_models_cuda", "tests", "models/cuda/tests"):
        for path in (ROOT / folder).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
                modules = []
                if isinstance(node, ast.Import):
                    modules = [name.name for name in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    modules = [node.module]
                for module in modules:
                    require(module.split(".")[0] not in forbidden, f"excluded import {module} in {path}")
                    if module.startswith("orbitinfer_models_cuda."):
                        relative = Path(*module.split("."))
                        base = ROOT / "models/cuda" / relative
                        require(base.is_dir() or base.with_suffix(".py").is_file(), f"missing model module {module} in {path}")
    for path in (ROOT / "crates").glob("*/Cargo.toml"):
        config = tomllib.loads(path.read_text())
        require(config["package"].get("publish") is False, f"publication is not disabled in {path}")
    require(not (ROOT / "models/mlx").exists(), "MLX execution is out of scope")
    require(not (ROOT / "models/torch").exists(), "CPU model execution is out of scope")
    print(f"Source boundary passed: {len(paths)} imported files; original license retained.")


if __name__ == "__main__":
    main()
