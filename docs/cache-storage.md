# Prefix cache storage

Prefix cache storage lets OrbitInfer reuse model state after it has been evicted from
GPU memory. The Engine manages prefix matching and the lifetime of device pages;
a storage provider keeps immutable checkpoints outside the
[device cache pool](cache-pool.md). OrbitInfer includes a bounded CPU DRAM provider, and
custom providers can implement other storage tiers through the same interface.

The [hierarchical external KV proposal](external-kv-design.md) specifies identity,
byte budgets and the planned OrbitKV/Mooncake Store/LMCache adapters. Identity,
budgets and independent operation/retirement receipts are implemented below. The [OrbitKV local DRAM adapter](orbitkv-dram.md) accepts fixed full-history CUDA
blocks. Its installed profile is unqualified; other adapters remain planned.

## Using the DRAM cache

Construct a provider and pass it to an Engine with prefix caching enabled. Given a
runner and prompt token IDs:

```python
from orbitinfer import CacheIdentity, Engine, MemoryCacheStorage, SamplingParams

storage = MemoryCacheStorage(max_bytes=8 * 1024**3)  # 8 GiB
try:
    with Engine(
        runner,
        prefix_cache=True,
        prefix_cache_storage=storage,
        prefix_cache_identity=CacheIdentity(
            model="model-id", weights="immutable-weight-revision",
            format="runner-cache-v1", dtype="bfloat16", shard="tp=0/1,pp=0/1",
            semantics="attention-rope-adapter-fingerprint",
        ),
        prefix_cache_max_bytes=256 * 1024**2,  # Outstanding borrowed views: 256 MiB
    ) as engine:
        request = engine.submit(
            token_ids,
            sampling_params=SamplingParams(max_output_tokens=128),
        )
        while not request.done:
            engine.step()
        output_tokens = request.take_result()
finally:
    storage.close()
```

`max_bytes` is required. The provider reserves its arena at construction, using
pinned host memory when CUDA is available. Its LRU budget charges object keys,
region names and payload bytes; general Python and allocator overhead is outside
that budget. One background worker owns a private CUDA copy stream per device.
Its queue defaults to 64 transfers (`max_inflight`), with borrowed bytes limited
to `max_bytes * max_inflight` (`max_pending_bytes` can set a smaller limit).
Rejection returns a false outcome with a successful no-access receipt.
It queues pending reads before fencing the stream and completing their futures.
Writes retire earlier reads before evicting or reusing their host memory.

The caller owns the provider. Close every Engine using it before calling
`storage.close()`. Stored objects can outlive an Engine, so compatible Engines can
reuse the same provider.

## How reuse works

OrbitInfer first looks for a usable checkpoint in the GPU prefix cache. If there is no
GPU hit, it asks the storage provider which checkpoint objects are available. A
storage hit is loaded into private device pages before the request resumes from
that checkpoint. The unmatched suffix, including any model-required replay
inputs, then runs normally. A partial GPU hit takes precedence over storage;
OrbitInfer does not search storage for a longer match in that case.
Requests for prompt logprobs run a full prefill instead of reusing checkpoints.

While a request waits for a load, the Engine can schedule other requests if cache
capacity permits. Transfers retain device pages, so memory pressure can still
limit dispatch. If an object has been evicted or a load fails, the request falls
back to prefill. Storage is an optional cache, so a successful presence check does
not guarantee that a later load will hit.

A checkpoint contains the state needed to resume at an aligned token boundary.
Full-history data is divided into objects for aligned spans, which can be shared
by several checkpoints. Sliding-window and recurrent state belong to a specific
checkpoint endpoint. Providers receive these objects as opaque keys and byte
regions; the Engine decides which combination forms a usable prefix.

## Request cache namespaces

Pass `cache_salt="project-a"` to `Engine.submit` to isolate prefix reuse for that
namespace. HTTP Chat Completions, Responses, and Anthropic Messages accept the
same top-level field. Reuse requires the same salt and matching text or media
identity, including when restoring checkpoints from a shared storage provider.
The salt changes cache keys, not prompt tokens or generated output.

Omitting `cache_salt` or passing `None` (`null` over HTTP) selects the unsalted
namespace. All external keys now include the representation identity; old
unbound keys intentionally miss. An empty string is a separate namespace.
Other values must be strings. Salts select cache namespaces; they do not provide
authentication or access control.

## Choosing a write policy

Set `prefix_cache_storage_policy` when constructing the Engine:

| Policy | When OrbitInfer offers a checkpoint to storage |
|---|---|
| `"write_through_selective"` (default) | When another request reuses that checkpoint from the GPU cache. |
| `"write_through"` | When the checkpoint is published, including its first use. |

With the selective policy, the first request creates a GPU checkpoint without
writing it to storage. A second request that reuses it triggers a backup of that
exact checkpoint. Later requests can restore the backup after GPU eviction.
Newly computed suffixes remain in the GPU cache until another request reuses
their checkpoints.

Selective writes avoid transferring prefixes that are used only once. Choose
`"write_through"` when prefixes are likely to be evicted from the GPU before their
first reuse: the selective policy cannot promote a checkpoint that is already
gone. Under either policy, providers skip copies of objects that remain resident
and coalesce duplicate pending writes. Eviction can make a later copy necessary.

## Concurrency and shutdown

`prefix_cache_max_inflight` bounds outstanding lookup, load and store **batches**
for one Engine. It counts operations, not keys or CUDA streams. Omitting it, or
passing `None`, uses `max(2, max_running_requests)`; an explicit value must be
positive. A larger limit permits more outstanding work and can retain more GPU
cache memory while transfers finish.

At the limit, new storage lookups and writes are skipped rather than queued by
the Engine. Requests that skip lookup proceed with prefill. A selective write
skipped at one GPU hit can be offered again on a later hit. The provider may
apply its own concurrency limits within the work it accepts.

Request completion or cancellation does not mean its storage transfers have
finished. The Engine keeps borrowed pages alive until the provider reports that
they are safe to recycle. Engine shutdown waits for those transfers; the provider
can continue to retain its own cached data afterward.

`engine.metrics_snapshot().cache["storage_pending"]` reports outstanding batches;
Prometheus exposes the same count as `orbitinfer:cache_storage_pending`. Restored prefix
lengths contribute to request cached-token metrics.

## Implementing a provider

The [`CacheStorage` protocol](../orbitinfer/cache_storage.py) has three methods.
Each returns promptly without waiting for I/O. Lookup returns a
`concurrent.futures.Future`; load and store return a `CacheTransfer` containing
two futures. These are ordinary methods, not `async def` methods.

| Method | Result |
|---|---|
| `lookup(keys)` | A tuple of presence booleans in key order. |
| `store(keys, buffers)` | `CacheTransfer`: publication outcome and borrowed-source retirement. |
| `load(keys, buffers)` | `CacheTransfer`: whole-load outcome and borrowed-destination retirement. |

Keys are opaque `bytes` values. For transfers, `keys` and `buffers` have equal
lengths, and `buffers[i]` maps string region names to borrowed, contiguous
`torch.uint8` tensors for `keys[i]`. Tensors may be on the CPU or GPU.
A store reads those views; a load writes into them. Match fields by name, not by
dictionary insertion order. An empty mapping is valid: it records a checkpoint
manifest with no payload.

Objects are immutable. Before reading source buffers, a provider must skip keys
it already holds and coalesce writes of the same pending key. The Engine may
offer an object more than once, including on repeated GPU cache hits.

### Keys and sharing

Keys identify token prefixes and checkpoint parts. They do not include a model
name, weight version or cache-layout fingerprint. The provider's instance or
backing-store namespace determines which Engines share data. Use separate
providers or partitions for incompatible model or cache representations.

### Transfer completion

`CacheTransfer(outcome, retirement)` separates the two facts:

- `outcome: Future[bool]` reports success (`True`), a miss/failure (`False`), or an
  operation exception. A successful load fills every requested region. A
  successful store publishes an immutable object that can be served completely.
- `retirement: Future[None]` succeeds with `None` only when all accesses to the
  borrowed buffers have ended. Pending, cancellation, an exception or a non-None
  result is not a release receipt. Failed retirement is sticky.

The Engine consumes an outcome only after successful retirement. A failed
outcome can arrive earlier without releasing pages. A retirement failure fences
new transfers to that storage binding and keeps the operation's exact exported
views, runner and page charges. Close reports failure; dropping the final Engine
reference still retains the unquiesced owners. A provider submission that raises
without returning a receipt is also conservatively unquiesced: it may have
accepted work before raising. Reject capacity before accepting access, then return
a failed outcome with successful retirement when nothing was posted.

Providers own and complete both futures. Consumers can inspect them but must not
complete them. Cancelling an outcome is not cancellation of DMA. For example,
the DRAM provider acknowledges a cancelled queued operation's retirement only
when its worker proves that no copy was started. `future.done()` alone never
authorizes buffer reuse. Providers must order GPU copies after their producer;
a separate CUDA stream alone does not establish that dependency.

For standalone use, `transfer.wait(timeout=...)` waits for both receipts within
one timeout budget and returns the bool outcome. It propagates operation errors
only after retirement and raises `CacheRetirementError` for an invalid or failed
retirement receipt. A timeout does not cancel, release or replay the transfer.
The Engine's normal polling uses the same receipts without blocking its scheduler.

A provider can release the device source after a proven copy into its own charged
staging allocation, then continue remote I/O against that copy. Retirement can
therefore precede publication. The Engine conservatively retains its operation
until both are observed; the provider cannot expose a partial object through
lookup. Its own staging allocation remains retained until the later I/O retires.
`MemoryCacheStorage.close()` raises `CacheRetirementError` after a failed fence;
repeated close does not retry that native cleanup.

## Runner support

Built-in CUDA runners implement
`cache_storage_buffers(groups, *, prefix_length)`. Custom runners return one
named-region mapping per supplied physical block, in group and block order.
These must be views of the bytes needed to resume that checkpoint; constructing
the views must not copy data. Guard-only blocks return `{}`. Include every byte
that a resumed kernel can read, even when that requires exporting a complete
intersecting page rather than only individual live rows.

The hook must accept empty groups with `prefix_length=0` and return no blocks.
The Engine uses this capability check during construction. Missing or
unsupported exporters raise `NotImplementedError` before resources are bound.

## Representation and transfer bounds

External storage requires `prefix_cache_identity=CacheIdentity(...)` and a positive
`prefix_cache_max_bytes`. The caller supplies authoritative model/weight, format,
stored dtype, semantic fingerprint and logical shard identity. Do not use mutable
weight aliases or physical GPU numbers. Identical replicas may share identity;
different tensor shards must not. The binding hashes this immutable value with
ordered cache group geometry, replay suffix and decode write span, then the
length-delimited request namespace and token chain. Arena capacity is excluded.
A mismatched identity misses before any destination copy; no implicit conversion
or resharding is performed.

All exports are validated before provider submission: contiguous byte views,
bounded names and region counts, total bytes and non-overlapping load destinations.
The limit is 65,536 keys/blocks/regions per operation and 1,024 UTF-8 bytes per name.
`prefix_cache_max_bytes` charges every exported view, including aliased read-only
sources, until **both** outcome and a successful retirement receipt are observed.
An oversized or over-budget operation posts nothing and falls back to prefill.
A failed receipt keeps its charge and fences the binding. Inspect
`engine.prefix_storage_stats` for current, peak and quarantined bytes, rejection
count and last preflight reason. Provider-resident DRAM has its separate budget.
