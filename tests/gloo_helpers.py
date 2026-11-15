"""Retire CPU-only fixture groups after their synchronous callers have stopped."""

import gc
from weakref import ref

from orbitinfer.backend.topology import RankGroup


def release_cpu_group_handles(*groups, workers=()):
    # All fixture collectives are synchronous, and callers join serve/watchers
    # before this boundary. Gloo's shutdown hook is a no-op; its destructor
    # joins runloop threads. A quarantined runner can keep these wrappers alive
    # after destroy_process_group, so detach only the retired fixture handles.
    wrappers = [
        value
        for group in groups
        for value in vars(group).values()
        if isinstance(value, RankGroup)
    ]
    handles = [ref(group.handle) for group in wrappers]
    for worker in workers:
        if hasattr(worker, "_wire"):
            worker._wire.group = None
    for group in wrappers:
        object.__setattr__(group, "handle", None)
    gc.collect()
    assert all(handle() is None for handle in handles), "fixture Gloo owner still live"
