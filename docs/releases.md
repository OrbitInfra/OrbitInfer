# Changes

## Unreleased

- External storage now requires `CacheIdentity` and `prefix_cache_max_bytes` at
  Engine construction. Keys include weights, format, dtype, semantics, shard and
  cache geometry; older keys intentionally miss. Invalid/overlapping exports and
  byte-limit rejection never call the provider. `prefix_storage_stats` reports
  borrowed and quarantined bytes. MemoryCacheStorage also bounds queued work.
- Native `EngineCore::new` now requires an `Arc` owning actual execution backing.
  Prepared or failed execution retains this owner on final drop. Python supplies
  its runner and retains uncertain completion/temporary owners. Successful close
  still releases them; no uncertain execution is replayed.

- External `CacheStorage.load` and `store` now return `CacheTransfer`, with
  separate `outcome` and `retirement` futures, instead of a bool/None future.
  Update providers to publish a successful retirement receipt only after all
  borrowed-buffer accesses have ended. A logical exception alone no longer
  releases pages. Use `transfer.wait(timeout=...)` for standalone blocking calls.
- `MemoryCacheStorage.close()` reports a failed device fence through
  `CacheRetirementError`; repeated close retains owners without retrying cleanup.
- `OrbitKVCacheStorage` adds source-level fixed-block GPU KV to local DRAM
  through the existing OrbitKV client. One full-history group is supported;
  variable window/state slices and SSD/peers are rejected or outside this profile.
  CPU shim controls are not hardware qualification. Explicit provider close is
  required; uncertain cleanup and unclosed registrations remain retained.
- Mooncake Store and LMCache adapters remain planned. See the
  [external KV design](external-kv-design.md) and [implementation plan](completion-plan.md).
