# OrbitKV local DRAM

`OrbitKVCacheStorage` maps fixed GPU KV blocks onto an existing same-host OrbitKV
Manager. OrbitInfer retains the existing prefix matcher, page allocator and
storage operation limits. OrbitKV owns DRAM residency, native registrations,
Publish/Restore and byte movement. The adapter issues no memcpy and does not add
an inference coordinator. No OrbitKV or OrbiTE source changes are required.

This is **implemented at source level; installed GPU profile unqualified**.
CPU tests drive the real adapter and Engine through an explicit CUDA/client shim,
including byte/guard checks. They are not a GPU or Manager data-path pass.

## Supported layout

The first profile is one full-history group, one block per physical slab, and a
single local GPU shard. Each registered layer is a contiguous CUDA `uint8` view
shaped `[physical_block_count, bytes_per_block]`; count includes slab zero. Layers
may have different byte widths but share the block count and device. The runner's
`cache_storage_buffers` must export exactly one whole block for every registered
layer, all at the same physical block index. No allocation or gather is needed
to form a contiguous byte view of an existing fixed block layout.

Window/state tails, multiple groups, packed slots, partial blocks, resharding and
CPU buffers are rejected. This does **not** yet integrate the full DeepSeek-V4.1
compressed/window/state layout. Its variable checkpoint slices need a separate
layout mapping; silently padding or concatenating them would change cache
semantics. SSD, peer and multi-shard profiles remain separate work.

## Binding

Install an OrbitKV client wheel exposing the native Save/Restore APIs. The source
contract was inspected at
[OrbitKV 93c1c55](https://github.com/OrbitInfra/OrbitKV/tree/93c1c55d795c19cb677f6661763d158a37a059af),
particularly `python/src/client.rs`, `python/src/local_restore.rs` and
`crates/orbitkv-channel/src/client.rs`. The adapter checks required methods before
registration. OrbitInfer does not import OrbitKV or Torch merely on import.

Configure the Manager for **local DRAM only**. Supply its bootstrap socket and
already allocated runner backing:

```python
from orbitinfer import CacheIdentity, Engine
from orbitinfer.integrations.orbitkv import OrbitKVCacheStorage

identity = CacheIdentity(
    model="model-id", weights="immutable-weight-revision", format="runner-cache-v1",
    dtype="bfloat16", shard="tp=0/1,pp=0/1", semantics="attention-rope-adapter-fingerprint",
)
# These byte views retain the existing GPU allocations. They include null slab 0.
layers = {"k": runner.key_blocks.view(torch.uint8).view(block_count, -1),
          "v": runner.value_blocks.view(torch.uint8).view(block_count, -1)}
with OrbitKVCacheStorage(
    "/run/orbitkv/orbitkv.sock", identity=identity, layers=layers,
    block_tokens=block_tokens, max_pending_bytes=256 * 1024**2,
) as storage:
    with Engine(runner, prefix_cache=True, prefix_cache_storage=storage,
                prefix_cache_identity=identity, prefix_cache_max_bytes=256 * 1024**2):
        ...  # Submit and drive requests through the usual Engine API.
```

Import `torch` at the caller. Names in `layers` must match the runner's exported
region names. One registration belongs to one Engine. Separate Engines with the
same representation use separate registrations and can share Manager objects.
The namespace includes representation identity, block width and layer names;
physical block capacity and process identity are excluded. Engine keys additionally
include its compiled cache geometry, namespace and token chain. A lookup is only
advisory: load claims a complete lease again before any GPU submission.

## Ownership

The adapter has one storage worker, at most 16 admitted operations by default
(configurable 1–64), a required byte limit, and bounded key/region metadata. This
worker bridges the existing blocking native client to `CacheTransfer`; it does
not schedule inference or move bytes. Producer events are recorded on the calling
stream and observed before Publish. Restore also passes that retained stream to
the native client's previous-user fence.

- `SaveNotSubmittedError` proves no Publish worker was launched and can retire a
  failed operation immediately.
- A native `done` result is the client's retirement receipt, including a drained
  failure. A pending result, exception or observation timeout is not a receipt.
- Unknown completion retains the client, registration, native handle, exact
  views and charges. It fences new submissions and is never replayed.
- Query leases are released if no Restore starts. Once Restore is attempted,
  the native client owns consumption; the adapter never guesses whether to
  release or replay it after an exception.
- Close the Engine before the provider. Provider close joins its bounded worker,
  then unregisters. Failed unregister/close is sticky and is never retried.
  Dropping a provider without successful explicit close conservatively retains
  its registration; use the context manager.

`timeout` defaults to 30 seconds (maximum 300) for observing a transfer, not for
releasing memory. No timer fabricates a retirement receipt. The Engine's separate
byte budget remains charged until it consumes the provider's successful receipt.

## Installed gate

On an explicitly authorized SM100 host with an installed OrbitKV Manager using
only DRAM, run:

```bash
uv run --no-sync python tests/hardware/orbitkv_dram.py \
  --bootstrap-socket /run/orbitkv/orbitkv.sock \
  --metrics-url http://127.0.0.1:9090/metrics \
  --device cuda:0 --output /tmp/orbitinfer-orbitkv-dram.json
```

The gate checks GPU bytes and guards, prefix restore through public Engine calls,
registration cleanup and route counters. It fails if SSD or peer byte counters
advance. Keep the report outside Git. It is written, not run; no full-model or
hardware qualification is claimed. See the [verification plan](completion-plan.md#installed-validation).
