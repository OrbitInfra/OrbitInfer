# Changes

## Unreleased

- External `CacheStorage.load` and `store` now return `CacheTransfer`, with
  separate `outcome` and `retirement` futures, instead of a bool/None future.
  Update providers to publish a successful retirement receipt only after all
  borrowed-buffer accesses have ended. A logical exception alone no longer
  releases pages. Use `transfer.wait(timeout=...)` for standalone blocking calls.
- `MemoryCacheStorage.close()` reports a failed device fence through
  `CacheRetirementError`; repeated close retains owners without retrying cleanup.
- No OrbitKV, Mooncake Store or LMCache adapter is shipped yet. See the
  [external KV design](external-kv-design.md) and [implementation plan](completion-plan.md).
