# Adapted for OrbitInfer; see NOTICE and docs/source-import.md.
"""Engine-owned prefix matching over asynchronous opaque byte storage."""

import hashlib
import json
import math
from collections.abc import Mapping
from concurrent.futures import Future
from dataclasses import asdict

import numpy as np

from orbitinfer._native import CacheGroupFamily
from orbitinfer.cache_storage import CacheIdentity, CacheTransfer

_MAX_REGIONS = 65536
_MAX_REGION_NAME = 1024


class _Operation:
    def __init__(self, owner, future, finish=lambda value: value, buffers=()):
        self.owner = owner
        self.future, self.finish, self.buffers = future, finish, buffers

    def poll(self, wait=False):
        if not wait and not self.future.done():
            return None
        try:
            value = self.future.result()
        except Exception:
            return 0 if self.future.done() else None
        return None if value is None else self.finish(value)


class _TransferOperation:
    """Retain actual exported views until the provider proves physical retirement."""

    def __init__(self, owner, buffers, charge):
        self.owner, self.buffers, self.charge = owner, buffers, charge
        self.transfer = None
        self.unquiesced = False

    def quarantine(self):
        if not self.unquiesced:
            self.owner.quarantined_bytes += self.charge
        self.unquiesced = True
        self.owner.halted = True

    def poll(self, wait=False):
        if self.unquiesced or self.transfer is None:
            return None
        receipt = self.transfer.retirement
        if not wait and not receipt.done():
            return None
        try:
            if receipt.result() is not None:
                self.quarantine()
                return None
        except BaseException:
            # A logical failure, cancelled future or failed fence is not a receipt.
            self.quarantine()
            return None
        outcome = self.transfer.outcome
        if not wait and not outcome.done():
            return None
        self.owner.borrowed_bytes -= self.charge
        self.charge = 0  # Repeated polls never refund the same owner twice.
        try:
            # Inspect failure without re-raising a stored exception: its traceback
            # would otherwise retain this operation and its retired buffer views.
            if outcome.cancelled() or outcome.exception() is not None:
                return 0
            result = outcome.result()
            return int(result is True)
        except BaseException:
            return 0  # Only a successful retirement receipt makes failure reusable.


class _PrefixStorage:
    def __init__(self, store, runner, layout, identity, max_bytes):
        self.runner = runner
        buffers = getattr(runner, "cache_storage_buffers", None)
        if not callable(buffers):
            raise NotImplementedError("prefix storage requires cache_storage_buffers")
        if buffers((), prefix_length=0):
            raise ValueError("empty cache export must return no blocks")
        self.store, self.buffers = store, buffers
        self.halted = False
        if layout is None:
            raise ValueError("prefix storage requires a managed cache layout")
        if not isinstance(identity, CacheIdentity):
            raise ValueError("prefix_cache_identity must be an explicit CacheIdentity")
        if type(max_bytes) is not int or not 0 < max_bytes <= (1 << 63) - 1:
            raise ValueError(
                "prefix_cache_max_bytes must be a positive signed 64-bit byte limit"
            )
        self.max_bytes = max_bytes
        self.borrowed_bytes = self.peak_bytes = self.quarantined_bytes = (
            self.rejected
        ) = 0
        self.last_rejection = ""
        self.groups = layout.groups
        geometry = {
            "groups": [
                [
                    g.block_tokens,
                    g.blocks_per_slab,
                    int(g.family),
                    int(g.retention),
                    g.window_tokens,
                    g.read_window_tokens,
                ]
                for g in self.groups
            ],
            "replay_suffix": layout.replay_suffix_tokens,
            "decode_extra": layout.decode_write_extra_tokens,
        }
        self.identity = hashlib.sha256(
            json.dumps(
                ["orbitinfer-cache-1", asdict(identity), geometry],
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).digest()
        self.alignment = math.lcm(*(g.block_tokens for g in self.groups))
        self.widths = tuple(self.alignment // g.block_tokens for g in self.groups)
        self.history = {
            i
            for i, g in enumerate(self.groups)
            if g.family != CacheGroupFamily.STATE
            and g.window_resident_block_count is None
        }
        validate = getattr(store, "validate_binding", None)
        if validate is not None:
            validate(identity, layout)

    def _keys(self, key):
        namespace, tokens = key
        encoded, width = np.asarray(tokens, dtype="<u4").tobytes(), self.alignment * 4
        if len(encoded) // width > _MAX_REGIONS:
            raise ValueError("cache key count exceeds metadata budget")
        digest, keys = hashlib.sha256(self.identity), []
        namespace = np.asarray(namespace or (), dtype="<u4").tobytes()
        digest.update(len(namespace).to_bytes(8, "little"))
        digest.update(namespace)
        for start in range(0, len(encoded) - width + 1, width):
            digest.update(encoded[start : start + width])
            keys.append(digest.digest())
        return keys

    def stats(self):
        return dict(
            limit_bytes=self.max_bytes,
            borrowed_bytes=self.borrowed_bytes,
            peak_bytes=self.peak_bytes,
            quarantined_bytes=self.quarantined_bytes,
            rejected=self.rejected,
            halted=self.halted,
            last_rejection=self.last_rejection,
        )

    def _reject(self, reason):
        self.rejected += 1
        self.last_rejection = reason
        future = Future()
        future.set_result(0)  # Preflight never called the provider.
        return _Operation(self, future)

    def lookup(self, key):
        if self.halted:
            future = Future()
            future.set_result(0)
            return _Operation(self, future)
        try:
            bases = self._keys(key)
        except ValueError as error:
            return self._reject(str(error))
        tags = ([b"h"] if self.history else []) + [b"t"]
        keys = tuple(key + tag for tag in tags for key in bases)

        def matched(mask):
            present, length = dict(zip(keys, mask, strict=True)), 0
            for i, key in enumerate(bases):
                if self.history and not present[key + b"h"]:
                    break
                if present[key + b"t"]:
                    length = (i + 1) * self.alignment
            return length

        return _Operation(self, self.store.lookup(keys), matched)

    def transfer(self, key, groups, load):
        if self.halted:
            future = Future()
            future.set_result(0)  # No exporter or provider call was made.
            return _Operation(self, future)
        try:
            keys, buffers, charge = self._export(key, groups, load)
        except (ValueError, TypeError, IndexError, StopIteration) as error:
            return self._reject(str(error))
        if charge > self.max_bytes - self.borrowed_bytes:
            return self._reject("borrowed cache byte budget exhausted")
        self.borrowed_bytes += charge
        self.peak_bytes = max(self.peak_bytes, self.borrowed_bytes)
        operation = _TransferOperation(self, buffers, charge)
        try:
            transfer = (self.store.load if load else self.store.store)(
                tuple(keys), buffers
            )
            if not isinstance(transfer, CacheTransfer):
                raise TypeError("storage load/store must return CacheTransfer")
            operation.transfer = transfer
        except BaseException:
            # The provider may have accepted a copy before raising. Keep the
            # exact views even if it did not return an operation handle.
            operation.quarantine()
        return operation

    def _export(self, key, groups, load):
        import torch

        _, tokens = key
        bases = self._keys(key)
        if not bases or len(groups) != len(self.groups):
            raise ValueError(
                "cache transfer requires an aligned checkpoint and every group"
            )
        if sum(len(rows) for _, rows in groups) > _MAX_REGIONS:
            raise ValueError("cache block count exceeds metadata budget")
        regions = iter(self.buffers(groups, prefix_length=len(tokens)))
        chunks, tail, intervals, charge, count = [{} for _ in bases], {}, {}, 0, 0
        for i, (first, rows) in enumerate(groups):
            for j in range(len(rows)):
                target = chunks[j // self.widths[i]] if i in self.history else tail
                exported = next(regions)
                if not isinstance(exported, Mapping):
                    raise TypeError("cache block must be a mapping of named regions")
                for name, region in exported.items():
                    count += 1
                    if count > _MAX_REGIONS:
                        raise ValueError("cache region count exceeds metadata budget")
                    if (
                        not isinstance(name, str)
                        or not 0 < len(name.encode()) <= _MAX_REGION_NAME
                    ):
                        raise ValueError(
                            "cache region name must be 1..1024 UTF-8 bytes"
                        )
                    if not isinstance(region, torch.Tensor):
                        raise TypeError("cache regions must be tensors")
                    if region.dtype != torch.uint8 or not region.is_contiguous():
                        raise ValueError("cache regions must be contiguous uint8 views")
                    # Charge each borrowed view, even if read-only sources alias.
                    charge += region.nbytes
                    if charge > self.max_bytes:
                        raise ValueError("cache transfer exceeds borrowed byte budget")
                    if load and region.nbytes:
                        intervals.setdefault(region.device, []).append(
                            (region.data_ptr(), region.data_ptr() + region.nbytes)
                        )
                    target[f"{i}/{first + j}/{name}"] = region
        if next(regions, None) is not None:
            raise ValueError("cache export returned too many blocks")
        for spans in intervals.values():
            spans.sort()
            if any(
                end > next_start for (_, end), (next_start, _) in zip(spans, spans[1:])
            ):
                raise ValueError("cache load destinations overlap")
        keys, buffers = (
            ([key + b"h" for key in bases], chunks) if self.history else ([], [])
        )
        keys.append(bases[-1] + b"t")
        buffers.append(tail)  # Empty tail still authenticates the checkpoint.
        return tuple(keys), tuple(buffers), charge
