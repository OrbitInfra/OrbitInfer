"""Unrun SM100 gate: public Engine, guarded GPU storage and final-owner containment."""

import argparse
import gc
from time import monotonic, sleep
from weakref import ref

import numpy as np
import torch

from orbitinfer import (
    CacheGroupLayout,
    CacheIdentity,
    CacheLayout,
    Engine,
    ForwardOutput,
    MemoryCacheStorage,
    RunnerConfig,
    SamplingParams,
)
from orbitinfer.backend.cuda.failures import PostAcceptCudaError


class Runner:
    runner_config = RunnerConfig(
        cache_layout=CacheLayout(parent_capacity=8, groups=[CacheGroupLayout(2)]),
        continuation_prefill=True,
    )

    def __init__(self, device, fill):
        self.arena = torch.full((9, 66), fill, dtype=torch.uint8, device=device)
        self.fail = False

    def bind_resources(self, _binding):
        pass

    def cache_storage_buffers(self, groups, *, prefix_length):
        return tuple(
            {"kv": self.arena[slab, 1:65]} for _, rows in groups for slab, _ in rows
        )

    def forward(self, plan):
        if self.fail:
            raise PostAcceptCudaError("injected uncertain execution")
        return ForwardOutput(
            token_ids=np.asarray(
                [int(plan.token_ids[end - 1]) + 1 for end in plan.token_indptr[1:]],
                dtype=np.uint32,
            ),
            token_indptr=np.arange(plan.num_sequences + 1, dtype=np.uint32),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if torch.cuda.get_device_capability(args.device)[0] != 10:
        raise RuntimeError("this installed profile requires SM100/Blackwell")
    identity = CacheIdentity("gate", "fixed", "raw-v1", "uint8", "tp=0/1", "plain")
    store = MemoryCacheStorage(4096)
    for fill in (73, 91):
        runner = Runner(args.device, fill)
        with Engine(
            runner,
            prefix_cache=True,
            prefix_cache_storage=store,
            prefix_cache_identity=identity,
            prefix_cache_max_bytes=512,
            prefix_cache_storage_policy="write_through",
        ) as engine:
            request = engine.submit(
                [1, 2, 3], sampling_params=SamplingParams(max_output_tokens=1)
            )
            deadline = monotonic() + 30
            while not request.done:
                if monotonic() > deadline:
                    raise TimeoutError("GPU cache transfer did not finish")
                engine.step()
                sleep(0.001)
            if fill == 91:
                assert request.cached_tokens == 2
                host = runner.arena.cpu()
                assert (host[:, (0, -1)] == 91).all()
                assert (host[:, 1:65] == 73).any()
            stats = engine.prefix_storage_stats
            assert stats is not None and stats["quarantined_bytes"] == 0
    store.close()
    runner = Runner(args.device, 91)
    runner.fail = True
    weak = ref(runner.arena)
    engine = Engine(runner)
    request = engine.submit([1], sampling_params=SamplingParams(max_output_tokens=1))
    try:
        engine.step()
    except PostAcceptCudaError:
        pass
    else:
        raise AssertionError("uncertain execution was reported as success")
    assert engine.poisoned and not request.done
    del runner, engine, request
    gc.collect()
    assert weak() is not None
    print("SM100 byte/guard and containment controls passed; no full-model claim")


if __name__ == "__main__":
    main()
