"""Fixed-block CUDA KV to a same-host OrbitKV Manager's DRAM tier.

The Manager must be configured for DRAM only for this profile. OrbitKV owns
registration, leases and byte movement. This adapter only translates immutable
objects and observes native receipts; it never copies KV bytes itself.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from threading import Lock
from time import monotonic, sleep
from uuid import uuid4

from orbitinfer import (
    CacheGroupFamily,
    CacheIdentity,
    CacheRetirementError,
    CacheTransfer,
)

# Failed native receipts keep the registration, client and exact borrowed views.
_RETAINED = []


@dataclass
class _Job:
    transfer: CacheTransfer
    buffers: tuple
    charge: int
    stream: object
    ready: object
    handle: object = None
    lease: bytes = b""


class OrbitKVCacheStorage:
    """One registration of fixed, contiguous CUDA blocks, shared by one Engine.

    `layers` maps exporter region names to contiguous uint8 tensors shaped
    [physical_blocks, bytes_per_block]. Include the Engine's reserved null slab.
    Only one full-history cache group with one block per slab is supported.
    No implicit gather/scatter, state/window conversion, SSD or peer profile is
    provided. `timeout` bounds receipt observation, never native lifetime.

    Close the Engine first, then this owner. A failed receipt or unregister is
    sticky CacheRetirementError and retains every owner; close never retries it.
    """

    def __init__(
        self,
        bootstrap_socket,
        *,
        identity,
        layers,
        block_tokens,
        max_pending_bytes,
        max_inflight=16,
        timeout=30.0,
    ):
        import torch

        if not isinstance(identity, CacheIdentity):
            raise ValueError("OrbitKV requires explicit CacheIdentity")
        if (
            type(block_tokens) is not int
            or block_tokens <= 0
            or type(max_inflight) is not int
            or not 0 < max_inflight <= 64
            or type(max_pending_bytes) is not int
            or not 0 < max_pending_bytes < 1 << 63
            or not 0 < timeout <= 300
        ):
            raise ValueError("invalid OrbitKV block, byte, operation or timeout budget")
        if not 0 < len(layers) <= 4096:
            raise ValueError("OrbitKV requires 1..4096 registered layers")
        self.layers = dict(sorted(layers.items()))
        for name, tensor in self.layers.items():
            if not isinstance(name, str) or not 0 < len(name.encode()) <= 1024:
                raise ValueError("invalid OrbitKV layer name")
            if (
                not isinstance(tensor, torch.Tensor)
                or not tensor.is_cuda
                or tensor.dtype != torch.uint8
                or tensor.ndim != 2
                or not tensor.is_contiguous()
                or min(tensor.shape) <= 0
            ):
                raise ValueError(
                    "OrbitKV requires contiguous 2D CUDA uint8 block views"
                )
        devices = {tensor.device for tensor in self.layers.values()}
        counts = {len(tensor) for tensor in self.layers.values()}
        if len(devices) != 1 or len(counts) != 1:
            raise ValueError(
                "OrbitKV layers must share a device and physical block count"
            )
        self.device, self.block_count = devices.pop(), counts.pop()
        spans = sorted(
            (t.data_ptr(), t.data_ptr() + t.nbytes) for t in self.layers.values()
        )
        if any(end > start for (_, end), (start, _) in zip(spans, spans[1:])):
            raise ValueError("OrbitKV registered layer views overlap")
        self.identity, self.block_tokens = identity, block_tokens
        self.max_pending_bytes, self.max_inflight, self.timeout = (
            max_pending_bytes,
            max_inflight,
            timeout,
        )
        self.pending_bytes = self.pending_count = 0
        self._lock, self._close_lock, self._jobs = Lock(), Lock(), {}
        self._bound = self._closed = self._retained = False
        self._error = None
        try:
            self.native = importlib.import_module("orbitkv")
        except ImportError as error:
            raise ImportError(
                "OrbitKV storage requires its optional native client wheel"
            ) from error
        for name in (
            "CacheManagerClient",
            "BlockHashes",
            "QueryReady",
            "QueryCandidates",
            "SaveNotSubmittedError",
        ):
            if not hasattr(self.native, name):
                raise RuntimeError(
                    f"OrbitKV client lacks {name}; see the pinned adapter contract"
                )
        for name in (
            "register_context_batch",
            "start_save",
            "wait_save",
            "start_restore",
            "wait_restore",
            "query_candidates",
            "query_prefetch",
            "cancel_query",
            "release",
            "unregister_context",
            "close",
        ):
            if not callable(getattr(self.native.CacheManagerClient, name, None)):
                raise RuntimeError(
                    f"OrbitKV client lacks {name}; see the pinned adapter contract"
                )
        self.namespace = (
            "orbitinfer-"
            + hashlib.sha256(
                json.dumps(
                    [
                        asdict(identity),
                        block_tokens,
                        [(n, t.shape[1]) for n, t in self.layers.items()],
                    ],
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
        )
        self.instance_id = "orbitinfer-" + uuid4().hex
        self.executor = ThreadPoolExecutor(1, thread_name_prefix="orbitinfer-orbitkv")
        self.client = self.native.CacheManagerClient(str(bootstrap_socket))
        try:
            self.registration = self.client.register_context_batch(
                self.instance_id,
                self.namespace,
                0,
                0,
                1,
                1,
                self.device.index,
                list(self.layers),
                [self.block_count] * len(self.layers),
                [tensor.shape[1] for tensor in self.layers.values()],
                [0] * len(self.layers),
                [1] * len(self.layers),
                "direct",
                False,
                layer_group_ids=[0] * len(self.layers),
                layer_formats=["exact"] * len(self.layers),
                tensors=list(self.layers.values()),
            )
        except BaseException as error:
            self._quarantine(error)
            self.executor.shutdown(wait=True)
            raise CacheRetirementError(
                "OrbitKV registration did not prove cleanup"
            ) from error

    def validate_binding(self, identity, layout):
        """Check the actual Engine geometry before admitting any cache operation."""
        if self._bound:
            raise ValueError("one OrbitKV registration belongs to one Engine")
        if identity != self.identity or len(layout.groups) != 1:
            raise ValueError("OrbitKV identity/group mismatch")
        group = layout.groups[0]
        if (
            group.family != CacheGroupFamily.HISTORY
            or group.window_resident_block_count is not None
            or group.blocks_per_slab != 1
            or group.block_tokens != self.block_tokens
            or layout.parent_capacity + 1 != self.block_count
        ):
            raise ValueError(
                "OrbitKV requires one fixed full-history block per physical slab"
            )
        self._bound = True

    @staticmethod
    def _roots(keys):
        keys = tuple(keys)
        if not 0 < len(keys) <= 65536 or any(
            not isinstance(k, bytes) or len(k) != 33 or k[-1:] not in (b"h", b"t")
            for k in keys
        ):
            raise ValueError("OrbitKV requires bounded Engine checkpoint keys")
        return keys, tuple(dict.fromkeys(k[:-1] for k in keys))

    def _reserve(self, charge):
        with self._lock:
            if self._closed or self._error:
                return False
            if (
                self.pending_count >= self.max_inflight
                or charge > self.max_pending_bytes - self.pending_bytes
            ):
                return False
            self.pending_count += 1
            self.pending_bytes += charge
            return True

    def _refund(self, charge):
        with self._lock:
            self.pending_count -= 1
            self.pending_bytes -= charge

    def _quarantine(self, error):
        with self._lock:
            if self._error is None:
                self._error = str(error) or type(error).__name__
            if not self._retained:
                _RETAINED.append(self)
                self._retained = True

    def lookup(self, keys):
        keys, roots = self._roots(keys)
        result = Future()
        if not self._reserve(0):
            result.set_result((False,) * len(keys))
            return result

        def run():
            try:
                if not result.set_running_or_notify_cancel():
                    return
                if self._error:
                    result.set_result((False,) * len(keys))
                    return
                answer = self.client.query_candidates(
                    self.instance_id,
                    self.native.BlockHashes(roots),
                    uuid4().hex,
                )
                hits = (
                    set(answer.hit_positions)
                    if isinstance(answer, self.native.QueryCandidates)
                    else set()
                )
                present = {root for i, root in enumerate(roots) if i in hits}
                result.set_result(tuple(key[:-1] in present for key in keys))
            except BaseException as error:
                result.set_exception(error)  # Advisory lookup cannot access GPU bytes.
            finally:
                self._refund(0)

        try:
            self.executor.submit(run)
        except BaseException as error:
            self._refund(0)
            result.set_exception(error)
        return result

    def _plan(self, keys, buffers):
        keys, _ = self._roots(keys)
        if len(buffers) > 65536 or sum(len(regions) for regions in buffers) > 65536:
            raise ValueError("OrbitKV region metadata budget exceeded")
        buffers = tuple(dict(regions) for regions in buffers)
        if len(keys) != len(buffers) or len(set(keys)) != len(keys):
            raise ValueError("OrbitKV key/region count mismatch or duplicate key")
        hashes, blocks, charge = [], [], 0
        for key, regions in zip(keys, buffers, strict=True):
            if key[-1:] == b"t":
                if regions:
                    raise ValueError(
                        "OrbitKV variable window/state tails are unsupported"
                    )
                continue
            found = {}
            for name, tensor in regions.items():
                if not isinstance(name, str) or len(name.encode()) > 1152:
                    raise ValueError("OrbitKV region name budget exceeded")
                group, logical, layer = name.split("/", 2)
                if (
                    group != "0"
                    or not logical.isdecimal()
                    or layer not in self.layers
                    or layer in found
                ):
                    raise ValueError("OrbitKV export does not match registered layers")
                registered = self.layers[layer]
                width = registered.shape[1]
                offset = tensor.data_ptr() - registered.data_ptr()
                if (
                    tensor.device != self.device
                    or tensor.dtype != registered.dtype
                    or not tensor.is_contiguous()
                    or tensor.nbytes != width
                    or offset < 0
                    or offset % width
                    or offset + width > registered.nbytes
                ):
                    raise ValueError("OrbitKV export is not one exact registered block")
                found[layer] = offset // width
                charge += width
                if charge > self.max_pending_bytes:
                    raise ValueError("OrbitKV borrowed byte budget exceeded")
            if found.keys() != self.layers.keys() or len(set(found.values())) != 1:
                raise ValueError("OrbitKV requires complete layers on the same block")
            hashes.append(key[:-1])
            blocks.append(next(iter(found.values())))
        if not hashes or len(set(blocks)) != len(blocks):
            raise ValueError("OrbitKV requires non-overlapping full-history blocks")
        return buffers, hashes, blocks, charge

    def store(self, keys, buffers):
        return self._submit(keys, buffers, load=False)

    def load(self, keys, buffers):
        return self._submit(keys, buffers, load=True)

    def _finish(self, job, success):
        with self._lock:
            self._jobs.pop(id(job), None)
        self._refund(job.charge)
        job.transfer.retirement.set_result(None)
        if not job.transfer.outcome.cancelled():
            job.transfer.outcome.set_result(success)

    def _submit(self, keys, buffers, *, load):
        import torch

        transfer = CacheTransfer(Future(), Future())
        transfer.retirement.set_running_or_notify_cancel()
        try:
            buffers, hashes, blocks, charge = self._plan(keys, buffers)
        except (ValueError, TypeError, AttributeError):
            transfer.outcome.set_result(False)
            transfer.retirement.set_result(None)
            return transfer
        if not self._reserve(charge):
            transfer.outcome.set_result(False)
            transfer.retirement.set_result(None)
            return transfer
        job = _Job(transfer, buffers, charge, None, None)
        with self._lock:
            self._jobs[id(job)] = job
        try:
            job.stream = torch.cuda.current_stream(self.device)
            job.ready = job.stream.record_event()
            self.executor.submit(self._run, job, hashes, blocks, load)
        except BaseException as error:
            self._fail(job, error)
        return transfer

    def _fail(self, job, error):
        self._quarantine(error)
        if not job.transfer.outcome.done():
            job.transfer.outcome.set_result(False)
        job.transfer.retirement.set_exception(
            CacheRetirementError(
                "OrbitKV did not prove retirement; registration, views and charges retained"
            )
        )

    def _run(self, job, hashes, blocks, load):
        try:
            if not job.transfer.outcome.set_running_or_notify_cancel():
                self._finish(job, False)
                return
            if self._error:
                self._finish(job, False)  # This queued job posted nothing.
                return
            deadline = monotonic() + self.timeout
            while not job.ready.query():
                if monotonic() >= deadline:
                    raise TimeoutError("producer event did not complete")
                sleep(0.001)
            if load:
                request_id = uuid4().hex
                while True:
                    answer = self.client.query_prefetch(
                        self.instance_id,
                        self.native.BlockHashes(hashes),
                        request_id,
                    )
                    if isinstance(answer, self.native.QueryReady):
                        break
                    if monotonic() >= deadline:
                        self.client.cancel_query(self.instance_id, request_id)
                        self._finish(job, False)  # Restore was never called.
                        return
                    sleep(0.001)
                job.lease = answer.lease
                if answer.num_hit_blocks != len(blocks) or set(
                    answer.hit_positions
                ) != set(range(len(blocks))):
                    if job.lease:
                        self.client.release(job.lease)
                    self._finish(job, False)
                    return
                job.handle = self.client.start_restore(
                    self.instance_id,
                    0,
                    self.device.index,
                    [list(self.layers)],
                    [(job.lease, [blocks])],
                    ready_stream=job.stream.cuda_stream,
                )
                status = self.client.wait_restore(
                    job.handle, timeout=max(0, deadline - monotonic())
                )
            else:
                try:
                    job.handle = self.client.start_save(
                        self.registration,
                        [(name, blocks, hashes) for name in self.layers],
                    )
                except self.native.SaveNotSubmittedError:
                    self._finish(job, False)
                    return
                status = self.client.wait_save(
                    job.handle, timeout=max(0, deadline - monotonic())
                )
            if type(status.done) is not bool or type(status.success) is not bool:
                raise CacheRetirementError("invalid native retirement status")
            if not status.done:
                raise CacheRetirementError(
                    status.message or "native operation remains pending"
                )
            self._finish(job, bool(status.success))
        except BaseException as error:
            self._fail(job, error)

    def close(self):
        with self._close_lock:
            self._close()

    def _close(self):
        with self._lock:
            self._closed = True
        self.executor.shutdown(wait=True)
        if self._error:
            raise CacheRetirementError(self._error)
        if not hasattr(self, "registration"):
            return
        try:
            success, message = self.client.unregister_context(self.instance_id)
            if not success:
                raise CacheRetirementError(message or "OrbitKV unregister failed")
            self.client.close()
            del self.registration
        except BaseException as error:
            self._quarantine(error)
            raise CacheRetirementError(
                "OrbitKV cleanup uncertain; never retried"
            ) from error

    def __del__(self):
        # Dropping a configured provider is not permission to deregister. The
        # explicit close path joins its bounded executor and receives cleanup.
        if hasattr(self, "registration"):
            self._quarantine("OrbitKV provider dropped without successful close")

    def __enter__(self):
        return self

    def __exit__(self, *_error):
        self.close()
