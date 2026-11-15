# Adapted for OrbitInfer; see NOTICE and docs/source-import.md.
"""Engine-owned prefix matching over asynchronous opaque byte storage."""

import hashlib
import math
from concurrent.futures import Future

import numpy as np

from orbitinfer._native import CacheGroupFamily
from orbitinfer.cache_storage import CacheTransfer


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

    def __init__(self, owner, buffers):
        self.owner, self.buffers = owner, buffers
        self.transfer = None
        self.unquiesced = False

    def quarantine(self):
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
    def __init__(self, store, runner, layout):
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
        self.groups = layout.groups
        self.alignment = math.lcm(*(g.block_tokens for g in self.groups))
        self.widths = tuple(self.alignment // g.block_tokens for g in self.groups)
        self.history = {
            i
            for i, g in enumerate(self.groups)
            if g.family != CacheGroupFamily.STATE
            and g.window_resident_block_count is None
        }

    def _keys(self, key):
        namespace, tokens = key
        encoded, width = np.asarray(tokens, dtype="<u4").tobytes(), self.alignment * 4
        digest, keys = hashlib.sha256(), []
        if namespace:
            digest.update(np.asarray(namespace, dtype="<u4").tobytes())
        for start in range(0, len(encoded) - width + 1, width):
            digest.update(encoded[start : start + width])
            keys.append((b"s" if namespace else b"") + digest.digest())
        return keys

    def lookup(self, key):
        if self.halted:
            future = Future()
            future.set_result(0)
            return _Operation(self, future)
        bases = self._keys(key)
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
        import torch

        tensor_type, byte_dtype = torch.Tensor, torch.uint8
        _, tokens = key
        bases = self._keys(key)
        regions = iter(self.buffers(groups, prefix_length=len(tokens)))
        chunks, tail = [{} for _ in bases], {}
        for i, (first, rows) in enumerate(groups):
            width = self.widths[i]
            for j in range(len(rows)):
                target = chunks[j // width] if i in self.history else tail
                name_prefix = f"{i}/{first + j}/"
                for name, region in next(regions).items():
                    if not isinstance(name, str) or not isinstance(region, tensor_type):
                        raise TypeError(
                            "cache regions must map string names to tensors"
                        )
                    if region.dtype != byte_dtype or not region.is_contiguous():
                        raise ValueError("cache regions must be contiguous uint8 views")
                    target[name_prefix + name] = region
        if next(regions, None) is not None:
            raise ValueError("cache export returned too many blocks")
        keys, buffers = ([], [])
        if self.history:
            keys, buffers = [key + b"h" for key in bases], chunks
        # Even an empty tail authenticates this exact checkpoint boundary.
        keys.append(bases[-1] + b"t")
        buffers.append(tail)
        buffers = tuple(buffers)
        operation = _TransferOperation(self, buffers)
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
