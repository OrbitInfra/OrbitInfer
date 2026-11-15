"""CPU client/CUDA shims exercise the production adapter; not a GPU profile."""

import gc
import sys
from concurrent.futures import Future
from threading import Event
from types import ModuleType, SimpleNamespace
from weakref import ref

import pytest
import torch
from test_cache_storage import IDENTITY

from orbitinfer import CacheGroupLayout, CacheLayout
from orbitinfer.cache_storage import CacheRetirementError
from orbitinfer.integrations.orbitkv import OrbitKVCacheStorage


class CudaView(torch.Tensor):
    @property
    def device(self):  # ty: ignore[invalid-property-type-override] -- CPU CUDA-claim shim
        return torch.device("cuda:0")

    @property
    def is_cuda(self):  # ty: ignore[invalid-property-type-override] -- CPU CUDA-claim shim
        return True


class QueryCandidates:
    def __init__(self, positions):
        self.hit_positions = positions


class QueryReady(QueryCandidates):
    def __init__(self, lease, positions):
        super().__init__(positions)
        self.lease, self.num_hit_blocks = lease, len(positions)


class SaveNotSubmittedError(RuntimeError):
    pass


@pytest.fixture
def native(monkeypatch):
    data, calls = {}, []
    gate = Future()
    gate.set_result(SimpleNamespace(done=True, success=True, message=""))
    state = SimpleNamespace(
        gate=gate, entered=Event(), ready=True, mode="ok", instances=[]
    )

    class Client:
        def __init__(self, socket):
            self.socket = socket
            self.layers = {}
            self.leases = {}
            state.instances.append(ref(self))

        def register_context_batch(
            self,
            instance,
            namespace,
            tp,
            pp,
            tp_size,
            world,
            device,
            names,
            counts,
            widths,
            strides,
            segments,
            backend,
            page_first,
            *,
            tensors,
            **metadata,
        ):
            calls.append(
                (
                    "register",
                    namespace,
                    tp,
                    pp,
                    tp_size,
                    world,
                    device,
                    backend,
                    page_first,
                )
            )
            assert strides == [0] * len(names) and segments == [1] * len(names)
            assert counts == [len(t) for t in tensors] and widths == [
                t.shape[1] for t in tensors
            ]
            assert metadata == dict(
                layer_group_ids=[0] * len(names), layer_formats=["exact"] * len(names)
            )
            self.layers = dict(zip(names, tensors, strict=True))
            self.namespace = namespace
            if state.mode == "register_fail":
                raise RuntimeError("uncertain registration")
            return object()

        def query_candidates(self, instance, hashes, req):
            calls.append(("lookup", tuple(hashes)))
            return QueryCandidates(
                [
                    i
                    for i, h in enumerate(hashes)
                    if all((self.namespace, name, h) in data for name in self.layers)
                ]
            )

        def query_prefetch(self, instance, hashes, req):
            answer = self.query_candidates(instance, hashes, req)
            self.leases[b"lease"] = hashes
            return QueryReady(b"lease", answer.hit_positions)

        def start_save(self, registration, saves):
            calls.append(("save", saves))
            if state.mode == "rejected":
                raise SaveNotSubmittedError("never submitted")
            if state.mode == "raise":
                raise RuntimeError("accepted before error")
            state.entered.set()
            return ("save", saves)

        def start_restore(self, instance, tp, device, groups, loads, *, ready_stream):
            calls.append(("restore", ready_stream, groups, loads))
            assert ready_stream == 73 and groups == [list(self.layers)]
            if state.mode == "restore_raise":
                self.leases.pop(loads[0][0])  # Native attempt consumed the lease.
                raise RuntimeError("restore accepted before error")
            state.entered.set()
            return ("load", loads)

        def wait_save(self, handle, *, timeout):
            return self.wait(handle, timeout)

        def wait_restore(self, handle, *, timeout):
            return self.wait(handle, timeout)

        def wait(self, handle, timeout):
            status = state.gate.result(timeout=timeout)
            if status.done and status.success:
                kind, tasks = handle
                if kind == "save":
                    for layer, blocks, hashes in tasks:
                        for block, h in zip(blocks, hashes, strict=True):
                            data[self.namespace, layer, h] = self.layers[layer][
                                block
                            ].clone()
                else:
                    for lease, (blocks,) in tasks:
                        for block, h in zip(
                            blocks, self.leases.pop(lease), strict=True
                        ):
                            for layer, tensor in self.layers.items():
                                tensor[block].copy_(data[self.namespace, layer, h])
            return status

        def release(self, lease):
            calls.append(("release", lease))
            self.leases.pop(lease)

        def cancel_query(self, instance, request):
            calls.append(("cancel_query", request))

        def unregister_context(self, instance):
            calls.append(("unregister", instance))
            if state.mode == "unregister_fail":
                return False, "uncertain unregister"
            self.layers.clear()
            return True, ""

        def close(self):
            calls.append(("close",))

    module = ModuleType("orbitkv")
    for name, value in dict(
        CacheManagerClient=Client,
        QueryReady=QueryReady,
        QueryCandidates=QueryCandidates,
        BlockHashes=tuple,
        SaveNotSubmittedError=SaveNotSubmittedError,
    ).items():
        setattr(module, name, value)
    monkeypatch.setitem(sys.modules, "orbitkv", module)
    stream = SimpleNamespace(
        cuda_stream=73, record_event=lambda: SimpleNamespace(query=lambda: state.ready)
    )
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: stream)
    return state, data, calls


def storage(*, identity=IDENTITY, max_pending_bytes=128, bind=True, **kwargs):
    layers = {
        name: torch.full((9, 8), 91, dtype=torch.uint8).as_subclass(CudaView)
        for name in ("k", "v")
    }
    owner = OrbitKVCacheStorage(
        "/fixture.sock",
        identity=identity,
        layers=layers,
        block_tokens=2,
        max_pending_bytes=max_pending_bytes,
        **kwargs,
    )
    if bind:
        owner.validate_binding(
            identity, CacheLayout(parent_capacity=8, groups=[CacheGroupLayout(2)])
        )
    return owner


def vector(owner, blocks=(1, 2)):
    keys = tuple(bytes([i]) * 32 + b"h" for i in range(len(blocks)))
    buffers = tuple(
        {f"0/{i}/{name}": tensor[block] for name, tensor in owner.layers.items()}
        for i, block in enumerate(blocks)
    )
    return keys + (keys[-1][:-1] + b"t",), buffers + ({},)


def test_gpu_block_contract_roundtrip_and_guards_without_adapter_byte_copies(native):
    source, target = storage(), storage()
    for tensor in source.layers.values():
        tensor[1:3].fill_(73)
    keys, buffers = vector(source)
    assert source.store(keys, buffers).wait(30)
    assert target.lookup(keys).result(30) == (True, True, True)
    _, targets = vector(target, (5, 6))
    assert target.load(keys, targets).wait(30)
    for tensor in target.layers.values():
        actual = tensor.as_subclass(torch.Tensor)
        assert (actual[5:7] == 73).all()
        assert (actual[:5] == 91).all() and (actual[7:] == 91).all()
    assert source.pending_bytes == target.pending_bytes == 0
    source.close()
    target.close()
    assert [row[0] for row in native[2]].count("save") == 1
    assert [row[0] for row in native[2]].count("restore") == 1


@pytest.mark.parametrize(
    "mode", ["pending", "raise", "unregister_fail", "register_fail"]
)
def test_unknown_receipt_or_cleanup_is_sticky_and_keeps_actual_owners(native, mode):
    state, _, calls = native
    if mode == "register_fail":
        state.mode = mode
        with pytest.raises(CacheRetirementError):
            storage()
        assert state.instances[-1]() is not None
        assert not any(c[0] in ("close", "unregister") for c in calls)
        return
    owner = storage()
    weak = ref(owner.layers["k"])
    state.mode = mode
    if mode == "pending":
        state.gate = Future()
        state.gate.set_result(
            SimpleNamespace(done=False, success=False, message="retained")
        )
    if mode != "unregister_fail":
        transfer = owner.store(*vector(owner))
        with pytest.raises(CacheRetirementError):
            transfer.wait(30)
        assert owner.pending_bytes == 32 and owner.pending_count == 1
        assert not owner.store(*vector(owner)).wait(30)
    for _ in range(2):
        with pytest.raises(CacheRetirementError):
            owner.close()
    assert [c[0] for c in calls].count("unregister") == int(mode == "unregister_fail")
    assert not any(c[0] == "close" for c in calls)
    del owner
    gc.collect()
    assert weak() is not None


def test_pending_save_holds_views_and_budget_until_native_drain(native):
    state, _, calls = native
    owner = storage(max_pending_bytes=32)
    state.gate = Future()
    keys, views = vector(owner)
    weak = ref(views[0]["0/0/k"])
    op = owner.store(keys, views)
    assert state.entered.wait(30)
    del views
    gc.collect()
    assert weak() is not None and not op.retirement.done()
    assert not owner.store(*vector(owner)).wait(0)
    assert owner.pending_bytes == 32
    state.gate.set_result(SimpleNamespace(done=True, success=True, message=""))
    assert op.wait(30)
    owner.close()
    gc.collect()
    assert weak() is None and owner.pending_bytes == 0
    assert [c[0] for c in calls].count("save") == 1


def test_no_access_rejection_and_invalid_layout_never_submit(native):
    state, _, calls = native
    owner = storage()
    state.mode = "rejected"
    assert not owner.store(*vector(owner)).wait(30)
    assert owner.pending_bytes == 0
    before = len(calls)
    keys, buffers = vector(owner)
    buffers[0]["0/0/k"] = owner.layers["k"][1, :4]
    assert not owner.load(keys, buffers).wait(0)
    assert len(calls) == before and owner.pending_bytes == 0
    with pytest.raises(ValueError, match="one Engine"):
        owner.validate_binding(
            IDENTITY, CacheLayout(parent_capacity=8, groups=[CacheGroupLayout(2)])
        )
    owner.close()


def test_queued_cancel_is_no_access_but_running_cancel_is_not_retirement(native):
    state, _, calls = native
    owner = storage()
    state.gate = Future()
    first = owner.store(*vector(owner))
    assert state.entered.wait(30)
    assert not first.outcome.cancel() and not first.retirement.done()
    second = owner.store(*vector(owner))
    assert second.outcome.cancel() and not second.retirement.done()
    assert owner.pending_count == 2 and owner.pending_bytes == 64
    state.gate.set_result(SimpleNamespace(done=True, success=True, message=""))
    assert first.wait(30)
    second.retirement.result(30)
    owner.close()
    assert owner.pending_bytes == owner.pending_count == 0
    assert [c[0] for c in calls].count("save") == 1


def test_close_joins_pending_copy_before_deregistering(native):
    from concurrent.futures import ThreadPoolExecutor

    state, _, calls = native
    owner = storage()
    state.gate = Future()
    operation = owner.store(*vector(owner))
    assert state.entered.wait(30)
    entered = Event()
    shutdown = owner.executor.shutdown

    def joining(*args, **kwargs):
        entered.set()
        return shutdown(*args, **kwargs)

    owner.executor.shutdown = joining
    with ThreadPoolExecutor(1) as worker:
        closed = worker.submit(owner.close)
        assert entered.wait(30)
        assert not closed.done() and not operation.retirement.done()
        assert not any(c[0] == "unregister" for c in calls)
        state.gate.set_result(SimpleNamespace(done=True, success=True, message=""))
        closed.result(30)
    assert operation.wait(0) and owner.pending_bytes == 0
    owner.close()  # Successful close is idempotent, failed close is sticky.
    assert [c[0] for c in calls].count("unregister") == 1


def test_missing_client_method_and_cpu_tensor_rejected_before_registration(
    native, monkeypatch
):
    from orbitinfer.integrations.orbitkv import OrbitKVCacheStorage

    with pytest.raises(ValueError, match="CUDA"):
        OrbitKVCacheStorage(
            "/fixture",
            identity=IDENTITY,
            layers={"x": torch.zeros((2, 8), dtype=torch.uint8)},
            block_tokens=2,
            max_pending_bytes=16,
        )
    assert not native[2]
    monkeypatch.setattr(sys.modules["orbitkv"].CacheManagerClient, "wait_restore", None)
    with pytest.raises(RuntimeError, match="wait_restore"):
        storage()
    assert not native[2]


def test_identity_namespace_cannot_load_other_weights(native):
    from dataclasses import replace

    source = storage()
    assert source.store(*vector(source)).wait(30)
    other = storage(identity=replace(IDENTITY, weights="another-checkpoint"))
    keys, buffers = vector(other)
    assert other.lookup(keys).result(30) == (False, False, False)
    assert not other.load(keys, buffers).wait(30)
    assert all((t.as_subclass(torch.Tensor) == 91).all() for t in other.layers.values())
    source.close()
    other.close()


@pytest.mark.parametrize("length", [3, 5, 7])
def test_real_engine_restores_only_after_the_native_client_receipt(
    native, monkeypatch, length
):
    from time import monotonic, sleep

    from test_flat_cache import RecordingCachedRunner

    from orbitinfer import Engine, SamplingParams

    for fill in (73, 91):
        owner = storage(bind=False)
        runner = RecordingCachedRunner(
            CacheLayout(parent_capacity=8, groups=[CacheGroupLayout(2)])
        )
        for tensor in owner.layers.values():
            tensor.fill_(fill)

        def export(groups, *, prefix_length):
            return tuple(
                {name: tensor[slab] for name, tensor in owner.layers.items()}
                for _, rows in groups
                for slab, _ in rows
            )

        monkeypatch.setattr(runner, "cache_storage_buffers", export, raising=False)
        with Engine(
            runner,
            prefix_cache=True,
            prefix_cache_storage=owner,
            prefix_cache_identity=IDENTITY,
            prefix_cache_max_bytes=128,
            prefix_cache_storage_policy="write_through",
        ) as engine:
            request = engine.submit(
                list(range(1, length + 1)),
                sampling_params=SamplingParams(max_output_tokens=1),
            )
            deadline = monotonic() + 30
            while not request.done:
                assert monotonic() < deadline
                engine.step()
                sleep(0.001)
            assert request.cached_tokens == (length - 1 if fill == 91 else 0)
            assert request.take_result() == [length + 1]
            if fill == 91:
                for tensor in owner.layers.values():
                    actual = tensor.as_subclass(torch.Tensor)
                    end = 1 + length // 2
                    assert (actual[1:end] == 73).all() and (actual[0] == 91).all()
                    assert (actual[end:] == 91).all()
        owner.close()


@pytest.mark.parametrize("status", ["pending", "restore_raise", "drained_failure"])
def test_restore_attempt_owns_lease_until_native_receipt_and_never_replays(
    native, status
):
    state, _, calls = native
    owner = storage()
    assert owner.store(*vector(owner)).wait(30)
    state.mode = status
    state.gate = Future()
    state.gate.set_result(
        SimpleNamespace(
            done=status == "drained_failure", success=False, message="injected restore"
        )
    )
    operation = owner.load(*vector(owner, (5, 6)))
    if status == "drained_failure":
        assert not operation.wait(30)
        assert owner.pending_bytes == owner.pending_count == 0
        owner.close()
    else:
        with pytest.raises(CacheRetirementError):
            operation.wait(30)
        assert owner.pending_bytes == 32 and owner.pending_count == 1
        assert not owner.load(*vector(owner, (5, 6))).wait(0)
        with pytest.raises(CacheRetirementError):
            owner.close()
        assert not any(c[0] == "unregister" for c in calls)
    assert [c[0] for c in calls].count("restore") == 1
    assert not any(c[0] == "release" for c in calls)
    assert all((t.as_subclass(torch.Tensor) == 91).all() for t in owner.layers.values())


def test_incomplete_restore_lease_released_before_any_gpu_submission(native):
    _, data, calls = native
    owner = storage()
    assert owner.store(*vector(owner)).wait(30)
    del data[owner.namespace, "v", bytes([1]) * 32]
    assert not owner.load(*vector(owner, (5, 6))).wait(30)
    assert [c[0] for c in calls].count("release") == 1
    assert not any(c[0] == "restore" for c in calls)
    assert owner.pending_bytes == owner.pending_count == 0
    assert all((t.as_subclass(torch.Tensor) == 91).all() for t in owner.layers.values())
    owner.close()
