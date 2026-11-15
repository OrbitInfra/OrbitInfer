"""Storage outcomes cannot retire a native Engine's actual borrowed views."""

import gc
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from threading import Event
from weakref import ref

import pytest
import torch
from test_cache_storage import make_engine

from orbitinfer import CacheRetirementError, CacheTransfer, SamplingParams


def completed(value):
    future = Future()
    future.set_result(value)
    return future


class HeldStorage:
    """Only the Engine operation retains the exported tensor objects."""

    def __init__(self, mode="transfer"):
        self.mode = mode
        self.transfer = CacheTransfer(Future(), Future())
        self.views = []
        self.loads = self.lookups = self.stores = 0

    def lookup(self, keys):
        self.lookups += 1
        return completed((True,) * len(keys))

    def load(self, keys, buffers):
        self.loads += 1
        self.views = [ref(region) for regions in buffers for region in regions.values()]
        if self.mode == "raise":
            raise RuntimeError("copy was accepted before submission raised")
        if self.mode == "legacy":
            # The old bool-future bridge interpreted this exception as a receipt.
            future = Future()
            future.set_exception(RuntimeError("logical failure before DMA retired"))
            return future
        return self.transfer

    def store(self, keys, buffers):
        self.stores += 1
        return CacheTransfer(completed(False), completed(None))


def pending_load(mode="transfer"):
    store = HeldStorage(mode)
    engine, runner, arena = make_engine(store, fill=99, limit=1)
    request = engine.submit(
        [1, 2, 3], sampling_params=SamplingParams(max_output_tokens=1)
    )
    engine.next_phase()
    assert store.loads == 1 and not runner.plans
    assert engine.metrics_snapshot().cache["storage_pending"] == 1
    assert store.views and all(view() is not None for view in store.views)
    return engine, runner, arena, request, store


@pytest.mark.parametrize("outcome", [True, False, "error", "cancel"])
def test_logical_outcome_does_not_retire_storage_pages(outcome):
    engine, runner, arena, request, store = pending_load()
    charged = engine.metrics_snapshot().cache["free_slabs"]
    before = arena.clone()
    if outcome == "error":
        store.transfer.outcome.set_exception(RuntimeError("failed but still writing"))
    elif outcome == "cancel":
        assert store.transfer.outcome.cancel()
    else:
        store.transfer.outcome.set_result(outcome)
    try:
        for _ in range(3):
            engine.next_phase()
        assert not runner.plans and request.cached_tokens == 0
        assert engine.metrics_snapshot().cache["storage_pending"] == 1
        assert engine.metrics_snapshot().cache["free_slabs"] == charged
        assert torch.equal(arena, before)
        with pytest.raises(TimeoutError):
            store.transfer.wait(0)
        assert all(view() is not None for view in store.views)
        # A late physical write remains legal until the independent receipt.
        touched = []
        for view in store.views:
            region = view()
            assert region is not None
            touched.append(region.data_ptr() - arena.data_ptr())
            region.fill_(73)
        del region
        store.transfer.retirement.set_result(None)
        engine.next_phase()
        assert engine.metrics_snapshot().cache["storage_pending"] == 0
        assert request.cached_tokens == (2 if outcome is True else 0)
        expected = before.clone()
        expected[touched] = 73
        assert torch.equal(
            arena, expected
        )  # Every byte outside the borrowed views is a guard.
        assert all(view() is None for view in store.views)
    finally:
        if not store.transfer.retirement.done():
            store.transfer.retirement.set_result(None)
        engine.close()


@pytest.mark.parametrize(
    "failure", ["exception", "cancelled", "invalid", "raise", "legacy"]
)
def test_uncertain_retirement_retains_exact_views_after_final_owner_drop(failure):
    mode = failure if failure in ("raise", "legacy") else "transfer"
    engine, runner, arena, request, store = pending_load(mode)
    if mode == "transfer":
        store.transfer.outcome.set_result(False)
        if failure == "exception":
            store.transfer.retirement.set_exception(RuntimeError("failed device fence"))
        elif failure == "cancelled":
            assert store.transfer.retirement.cancel()
        else:
            store.transfer.retirement.set_result(
                True
            )  # Not a valid retirement receipt.
    engine.next_phase()
    charged = engine.metrics_snapshot().cache["free_slabs"]
    assert engine.cancel(request)
    engine.clear_prefix_cache()
    engine.submit([8, 9, 10], sampling_params=SamplingParams(max_output_tokens=1))
    engine.step()
    # No new calls to the uncertain storage owner, including optional backups.
    assert (store.lookups, store.loads, store.stores) == (1, 1, 0)
    assert engine.metrics_snapshot().cache["storage_pending"] == 1
    assert engine.metrics_snapshot().cache["free_slabs"] <= charged
    assert arena.tolist() == [99] * len(arena)
    with pytest.raises(RuntimeError):
        engine.close()
    assert engine.poisoned
    retained_runner, retained_arena = ref(runner), ref(arena)
    del request, runner, arena, engine
    gc.collect()
    assert retained_runner() is not None and retained_arena() is not None
    assert all(view() is not None for view in store.views)
    assert all(view().tolist() == [99] for view in store.views)


def test_cancel_and_close_wait_for_storage_retirement():
    engine, _, arena, request, store = pending_load()
    assert engine.cancel(request)
    store.transfer.outcome.set_result(False)
    entered = Event()

    original_result = store.transfer.retirement.result

    def wait_receipt(timeout=None):
        entered.set()
        return original_result(timeout)

    store.transfer.retirement.result = wait_receipt
    with ThreadPoolExecutor(1) as worker:
        closing = worker.submit(engine.close)
        try:
            assert entered.wait(30)
            assert not closing.done()
            assert all(view() is not None for view in store.views)
            assert arena.tolist() == [99] * len(arena)
        finally:
            store.transfer.retirement.set_result(None)
        closing.result(30)
    assert engine.closed and not engine.poisoned
    assert engine.metrics_snapshot().cache["storage_pending"] == 0
    assert all(view() is None for view in store.views)


def test_last_engine_reference_drains_before_releasing_the_arena():
    engine, runner, arena, request, store = pending_load()
    store.transfer.outcome.set_result(False)
    entered, returned = Event(), Event()
    original_result = store.transfer.retirement.result

    def wait_receipt(timeout=None):
        entered.set()
        return original_result(timeout)

    store.transfer.retirement.result = wait_receipt
    retained_runner, retained_arena = ref(runner), ref(arena)
    holder = [engine]
    del engine, runner, arena, request

    def drop():
        holder.clear()
        returned.set()

    with ThreadPoolExecutor(1) as worker:
        dropping = worker.submit(drop)
        try:
            assert entered.wait(30)
            assert not returned.is_set()
            assert retained_runner() is not None and retained_arena() is not None
            assert all(view() is not None for view in store.views)
            assert retained_arena().tolist() == [99] * 33
        finally:
            store.transfer.retirement.set_result(None)
        dropping.result(30)
    gc.collect()
    assert returned.is_set()
    assert retained_runner() is None and retained_arena() is None
    assert all(view() is None for view in store.views)


@pytest.mark.parametrize("invalid", [Future(), None, True])
def test_transfer_constructor_requires_two_distinct_futures(invalid):
    with pytest.raises((TypeError, ValueError)):
        CacheTransfer(invalid, invalid)


def test_failed_retirement_is_never_reported_as_a_safe_result():
    receipt = Future()
    receipt.set_exception(RuntimeError("unconfirmed native cleanup"))
    transfer = CacheTransfer(completed(True), receipt)
    for _ in range(2):
        with pytest.raises(CacheRetirementError):
            transfer.wait()
