"""Unrun SM100 GPU KV / local OrbitKV DRAM gate. Start a DRAM-only Manager first."""

import argparse
import json
from importlib.metadata import version
from pathlib import Path
from time import monotonic, sleep
from urllib.request import urlopen

import numpy as np
import torch

from orbitinfer import (
    CacheGroupLayout,
    CacheIdentity,
    CacheLayout,
    Engine,
    ForwardOutput,
    RunnerConfig,
    SamplingParams,
)
from orbitinfer.integrations.orbitkv import OrbitKVCacheStorage


class Runner:
    runner_config = RunnerConfig(
        cache_layout=CacheLayout(parent_capacity=8, groups=[CacheGroupLayout(2)]),
        continuation_prefill=True,
    )

    def __init__(self, device, fill, width):
        self.layers = {
            name: torch.full((9, width), fill, device=device, dtype=torch.uint8)
            for name in ("k", "v")
        }

    def bind_resources(self, _binding):
        pass

    def cache_storage_buffers(self, groups, *, prefix_length):
        return tuple(
            {name: tensor[slab] for name, tensor in self.layers.items()}
            for _, rows in groups
            for slab, _ in rows
        )

    def forward(self, plan):
        return ForwardOutput(
            token_ids=np.asarray(
                [int(plan.token_ids[end - 1]) + 1 for end in plan.token_indptr[1:]],
                dtype=np.uint32,
            ),
            token_indptr=np.arange(plan.num_sequences + 1, dtype=np.uint32),
        )


def metrics(url):
    result = {}
    with urlopen(url, timeout=10) as response:
        for line in response.read(8 << 20).decode().splitlines():
            if line and not line.startswith("#"):
                name, value = line.rsplit(" ", 1)
                name = name.split("{", 1)[0]
                result[name] = result.get(name, 0) + float(value)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-socket", required=True)
    parser.add_argument("--metrics-url", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if torch.cuda.get_device_capability(args.device)[0] != 10:
        raise RuntimeError("this installed profile requires SM100/Blackwell")
    before = metrics(args.metrics_url)
    checks = []
    for width in (4096, 65536):
        identity = CacheIdentity(
            "installed-gate",
            "fixed",
            f"bytes-{width}",
            "uint8",
            "tp=0/1,pp=0/1",
            "full-history",
        )
        for fill in (73, 91):
            runner = Runner(args.device, fill, width)
            with OrbitKVCacheStorage(
                args.bootstrap_socket,
                identity=identity,
                layers=runner.layers,
                block_tokens=2,
                max_pending_bytes=8 * width,
            ) as storage:
                with Engine(
                    runner,
                    prefix_cache=True,
                    prefix_cache_storage=storage,
                    prefix_cache_identity=identity,
                    prefix_cache_max_bytes=8 * width,
                    prefix_cache_storage_policy="write_through",
                ) as engine:
                    request = engine.submit(
                        [1, 2, 3], sampling_params=SamplingParams(max_output_tokens=1)
                    )
                    deadline = monotonic() + 60
                    while not request.done:
                        if monotonic() > deadline:
                            raise TimeoutError(
                                "OrbitKV Engine transfer did not complete"
                            )
                        engine.step()
                        sleep(0.001)
                    if request.take_result() != [4]:
                        raise AssertionError("wrong continuation output")
                    if fill == 91:
                        if request.cached_tokens != 2:
                            raise AssertionError("no OrbitKV prefix restore")
                        for tensor in runner.layers.values():
                            actual = tensor.cpu()
                            if not (
                                (actual[1] == 73).all()
                                and (actual[0] == 91).all()
                                and (actual[2:] == 91).all()
                            ):
                                raise AssertionError("payload or guard mismatch")
                if storage.pending_count or storage.pending_bytes:
                    raise AssertionError("drained Engine still has a charged transfer")
            checks.append(dict(width=width, fill=fill, bytes_verified=True))
    after = metrics(args.metrics_url)
    for name in ("orbitkv_save_bytes_total", "orbitkv_load_bytes_total"):
        if name not in before or name not in after or after[name] <= before[name]:
            raise AssertionError(f"missing native route evidence: {name}")
    for name in set(before) | set(after):
        if ("ssd" in name or "remote" in name) and "bytes" in name:
            if after.get(name, 0) > before.get(name, 0):
                raise AssertionError(f"non-DRAM route observed: {name}")
    args.output.write_text(
        json.dumps(
            dict(
                orbitinfer=version("orbitinfer"),
                orbitkv=version("orbitkv"),
                torch=torch.__version__,
                device=torch.cuda.get_device_name(args.device),
                checks=checks,
                metrics_before=before,
                metrics_after=after,
                scope="GPU KV/local DRAM; no full model or SSD/peer claim",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
