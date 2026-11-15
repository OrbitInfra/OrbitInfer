"""Representation authority and borrowed-byte admission over real CPU views."""

from dataclasses import fields, replace
from types import SimpleNamespace

import pytest
import torch
from test_cache_retirement import HeldStorage
from test_cache_storage import IDENTITY, make_engine

from orbitinfer import CacheGroupLayout, CacheLayout, SamplingParams
from orbitinfer._cache_storage import _PrefixStorage


def binding(
    store=None, *, identity=IDENTITY, capacity=8, block=2, limit=16, exporter=None
):
    arena = torch.arange(32, dtype=torch.uint8)

    def buffers(groups, *, prefix_length):
        return tuple(
            {"bytes": arena[row : row + 1]} for _, rows in groups for row, _ in rows
        )

    runner = SimpleNamespace(cache_storage_buffers=exporter or buffers)
    return _PrefixStorage(
        store or HeldStorage(),
        runner,
        CacheLayout(parent_capacity=capacity, groups=[CacheGroupLayout(block)]),
        identity,
        limit,
    )


def test_identity_includes_every_authoritative_field_and_geometry():
    key = ([], [1, 2, 3, 4])
    original = binding()._keys(key)
    for field in fields(IDENTITY):
        changed = replace(IDENTITY, **{field.name: "different"})
        assert set(original).isdisjoint(binding(identity=changed)._keys(key))
    assert set(original).isdisjoint(binding(block=4)._keys(key))
    assert (
        binding(capacity=64)._keys(key) == original
    )  # Physical capacity is not format.
    assert set(original).isdisjoint(binding()._keys(([1], key[1])))


@pytest.mark.parametrize("invalid", [None, "", 1, "x" * 1025])
def test_invalid_identity_and_unbounded_budget_rejected_before_io(invalid):
    with pytest.raises(ValueError, match="cache identity weights"):
        replace(IDENTITY, weights=invalid)
    store = HeldStorage()
    for identity, limit in [
        (None, 16),
        (IDENTITY, 0),
        (IDENTITY, -1),
        (IDENTITY, 1 << 63),
    ]:
        with pytest.raises(ValueError, match="prefix_cache"):
            binding(store, identity=identity, limit=limit)
    assert (store.lookups, store.loads, store.stores) == (0, 0, 0)


@pytest.mark.parametrize("failure", [False, True])
def test_byte_charge_survives_logical_completion_and_failed_receipt(failure):
    store = HeldStorage()
    owner = binding(store, limit=1)
    key, groups = ([], [1, 2]), ((0, ((1, 0),)),)
    first = owner.transfer(key, groups, True)
    assert owner.stats()["borrowed_bytes"] == 1
    rejected = owner.transfer(key, groups, True)
    assert rejected.poll() == 0 and store.loads == 1
    store.transfer.outcome.set_result(False)
    assert first.poll() is None and owner.borrowed_bytes == 1
    if failure:
        store.transfer.retirement.set_exception(RuntimeError("uncertain fence"))
        assert first.poll() is None
        assert owner.halted and owner.quarantined_bytes == owner.borrowed_bytes == 1
        assert owner.transfer(key, groups, True).poll() == 0 and store.loads == 1
    else:
        store.transfer.retirement.set_result(None)
        assert first.poll() == first.poll() == 0
        assert owner.borrowed_bytes == 0 and owner.peak_bytes == 1
        assert owner.transfer(key, groups, True).poll() == 0 and store.loads == 2
        assert owner.borrowed_bytes == 0


@pytest.mark.parametrize(
    "case", ["oversize", "overlap", "dtype", "noncontiguous", "too_many"]
)
def test_invalid_exports_never_reach_provider_or_touch_guards(case):
    arena = torch.full((16,), 91, dtype=torch.uint8)
    store = HeldStorage()
    region = {"a": arena[2:6]}
    if case == "overlap":
        region["b"] = arena[3:7]
    elif case == "dtype":
        region = {"a": arena.float()}
    elif case == "noncontiguous":
        region = {"a": arena[::2]}

    def exporter(groups, *, prefix_length):
        return (region,) * (2 if case == "too_many" else 1) if groups else ()

    owner = binding(store, exporter=exporter, limit=1 if case == "oversize" else 64)
    op = owner.transfer(([], [1, 2]), ((0, ((1, 0),)),), True)
    assert op.poll() == 0 and store.loads == 0
    assert owner.rejected == 1 and owner.borrowed_bytes == 0
    assert arena.tolist() == [91] * 16


def test_native_storage_admission_consumes_the_byte_budget_and_restores_it():
    store = HeldStorage()
    engine, runner, arena = make_engine(store, max_bytes=1)
    first = engine.submit(
        [1, 2, 3], sampling_params=SamplingParams(max_output_tokens=1)
    )
    engine.next_phase()
    second = engine.submit(
        [4, 5, 6], sampling_params=SamplingParams(max_output_tokens=1)
    )
    engine.next_phase()
    assert store.loads == 1 and not runner.plans
    assert engine.prefix_storage_stats["borrowed_bytes"] == 1
    assert engine.prefix_storage_stats["rejected"] == 1
    assert arena.tolist() == [0] * len(arena)
    assert engine.cancel(first) and engine.cancel(second)
    store.transfer.outcome.set_result(False)
    store.transfer.retirement.set_result(None)
    engine.close()
    assert engine.prefix_storage_stats["borrowed_bytes"] == 0
    assert engine.metrics_snapshot().cache["storage_pending"] == 0


def test_reference_provider_bounds_queued_jobs_and_bytes_before_enqueue(monkeypatch):
    from threading import Event

    from orbitinfer import MemoryCacheStorage

    store = MemoryCacheStorage(4096, max_inflight=1, max_pending_bytes=4)
    entered, retire = Event(), Event()
    original = store._copy

    def copy(pairs, load):
        entered.set()
        assert retire.wait(30)
        original(pairs, load)

    monkeypatch.setattr(store, "_copy", copy)
    source = torch.tensor([1, 2, 3, 4], dtype=torch.uint8)
    first = store.store((b"first",), ({"x": source},))
    try:
        assert entered.wait(30)
        assert not store.store((b"second",), ({"x": source},)).wait(0)
        assert (store.pending_count, store.pending_bytes) == (1, 4)
    finally:
        retire.set()
    assert first.wait(30)
    assert (store.pending_count, store.pending_bytes) == (0, 0)
    assert not store.store(
        (b"large",), ({"x": torch.zeros(5, dtype=torch.uint8)},)
    ).wait(0)
    store.close()
