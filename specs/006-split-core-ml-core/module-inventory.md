# Core Module Migration Inventory

| Original surface | Destination | Status |
|------------------|-------------|--------|
| `aitraf_core.cache` | `aitraf_core.cache` | Retained and tested |
| `aitraf_core.utils.jsonl` | `aitraf_core.utils.jsonl` | Retained and tested |
| `aitraf_core.storage.s3` | `aitraf_core.storage.s3` | Retained for API and train |
| `aitraf_core.storage.clips` | `aitraf_train.storage.clips` | Moved; old path removed |
| `aitraf_core.utils.huggingface` | `aitraf_ml_core.utils.huggingface` | Moved; old path removed |
| `aitraf_core.inference` and descendants | `aitraf_ml_core.inference` | Moved; internal and consumer imports updated |
| `aitraf_core.loading` and descendants | `aitraf_ml_core.loading` | Moved; tests migrated |
| `aitraf_core.pre_processing` and descendants | `aitraf_ml_core.pre_processing` | Moved; core cache import retained |
| `aitraf_core.processing` and descendants | `aitraf_ml_core.processing` | Moved; internal and consumer imports updated |

All 29 original source files are covered by the grouped tree entries above.
Search and import-spec validation confirms that no forwarding module remains.
