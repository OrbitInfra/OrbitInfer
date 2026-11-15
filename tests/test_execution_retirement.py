"""Final-reference controls use actual backing, not only logical page counts."""

import gc
from weakref import ref

import numpy as np
import pytest
from output_helpers import forward_output
from test_flat_cache import RecordingCachedRunner

from orbitinfer import CacheGroupLayout, CacheLayout, Engine, SamplingParams
from orbitinfer.backend.cuda.failures import PostAcceptCudaError


class BackedRunner(RecordingCachedRunner):
    def __init__(self, layout):
        super().__init__(layout)
        self.backing = np.full(64, 41, dtype=np.uint8)


@pytest.mark.parametrize(
    "failure",
    ["forward", "inspection", "harvest", "feedback", "next_enqueue", "next_inspection"],
)
def test_final_engine_drop_retains_runner_and_uncertain_completion(failure):
    references = []

    class Completion:
        def __init__(self, serial):
            self.serial = serial
            self.backing = np.full(64, 73, dtype=np.uint8)
            references.append(ref(self.backing))

        @property
        def result(self):
            if failure == "inspection" or (
                failure == "next_inspection" and self.serial == 2
            ):
                raise RuntimeError("uncertain inspection")
            return self.harvest

        @property
        def feedback(self):
            if failure == "feedback":
                temporary = np.full(64, 73, dtype=np.uint8)
                references.append(ref(temporary))
                raise RuntimeError("uncertain feedback")
            return object()

        def harvest(self):
            if failure != "harvest":
                return forward_output([3])
            temporary = np.full(64, 73, dtype=np.uint8)
            references.append(ref(temporary))
            raise RuntimeError("uncertain harvest")

    class Model(BackedRunner):
        calls = 0

        def forward(self, plan):
            if failure == "forward":
                temporary = np.full(64, 73, dtype=np.uint8)
                references.append(ref(temporary))
                raise PostAcceptCudaError("uncertain forward")
            return super().forward(plan)

        def enqueue(self, _plan, *, feedback=None):
            self.calls += 1
            if failure == "next_enqueue" and self.calls == 2:
                temporary = np.full(64, 73, dtype=np.uint8)
                references.append(ref(temporary))
                raise RuntimeError("uncertain continuation enqueue")
            return Completion(self.calls)

    runner = Model(CacheLayout(parent_capacity=4, groups=[CacheGroupLayout(128)]))
    backing, owner = ref(runner.backing), ref(runner)
    engine = Engine(runner)
    request = engine.submit([1], sampling_params=SamplingParams(max_output_tokens=10))
    if failure != "forward":
        engine.step()  # Prefill retires normally.
        if failure in ("harvest", "feedback", "next_enqueue", "next_inspection"):
            engine.step()  # Enqueue the owned completion.
    with pytest.raises(RuntimeError, match="uncertain"):
        engine.step(allow_overlap=failure != "harvest")
    assert engine.poisoned and not request.done
    assert engine.cache_occupied_block_count == 1 and not runner.released
    with pytest.raises(RuntimeError):
        engine.close()
    del engine, request, runner
    gc.collect()
    assert owner() is not None and backing() is not None
    assert np.all(backing() == 41)
    assert all(item() is not None and np.all(item() == 73) for item in references)


def test_successful_close_releases_actual_runner_backing():
    runner = BackedRunner(CacheLayout(parent_capacity=4, groups=[CacheGroupLayout(2)]))
    backing, owner = ref(runner.backing), ref(runner)
    engine = Engine(runner)
    request = engine.submit([1], sampling_params=SamplingParams(max_output_tokens=1))
    engine.step()
    assert request.done
    engine.close()
    assert engine.cache_occupied_block_count == 0
    del engine, request, runner
    gc.collect()
    assert owner() is None and backing() is None
