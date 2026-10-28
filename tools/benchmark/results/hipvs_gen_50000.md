# hipVS HNSW build benchmark

Generated: 2026-10-28 18:39:41 UTC

## Environment

- server_version: `OceanBase 4.3.5.3 seekdb (r1.4.0.0) (Built 1-6cc4b91ae856856e4041c3144609f39bc5f2143d Oct 28 2026)`
- cpu: `AMD EPYC 9334 32-Core Processor`
- cpu_quota: `1600000 100000`
- nproc: `128`
- memory: `503 GiB`
- gpu: `Card Series: 		AMD Radeon Graphics;GFX Version: 		gfx1100`
- vram: `47.9844 GiB`
- rocm: `7.2.4`
- hipvs: `1.0.0`
- uname: `Linux 6.8.0-79-generic`
- host: `127.0.0.1`
- port: `2884`

## Configuration

- rows: `50000`
- dim: `128`
- queries: `100`
- topk: `10`
- metric: `l2`
- m: `16`
- ef_construction: `200`
- ef_search: `64`
- runs: `3`
- rounds: `1`
- dataset: `generated standard normal, seed 20261028`
- index DDL: `CREATE VECTOR INDEX idx_vec ON `bench_<lib>` (vec) WITH (DISTANCE=l2, TYPE=HNSW, LIB=<LIB>, M=16, EF_CONSTRUCTION=200, EF_SEARCH=64)`

## Separately recorded preparation

| step | seconds |
| --- | --- |
| dataset load / generation + exact truth (client) | 0.33 |
| base table import (no index) | 2.47 |
| copy into bench_vsag (no index) | 0.26 |
| copy into bench_hipvs (no index) | 0.26 |

## Build: CREATE VECTOR INDEX until the first query returns

Each run starts from a table with data and no vector index. `create` is the DDL statement, `first query` the first APPROXIMATE query afterwards; `build` = both.

| LIB | run | create (s) | first query (s) | build (s) | GPU path (observer log) | kNN / optimize / load (ms) | CPU fallback |
| --- | --- | --- | --- | --- | --- | --- | --- |
| vsag | 1 | 15.003 | 0.034 | 15.036 | - | - | n/a |
| vsag | 2 | 15.290 | 0.035 | 15.325 | - | - | n/a |
| vsag | 3 | 15.196 | 0.034 | 15.229 | - | - | n/a |
| hipvs | 1 | 0.839 | 0.028 | 0.867 | exact_knn | 304.8 / 93.2 / 1.6 | no |
| hipvs | 2 | 0.763 | 0.020 | 0.784 | exact_knn | 267.1 / 91.3 / 1.6 | no |
| hipvs | 3 | 0.834 | 0.020 | 0.854 | exact_knn | 268.4 / 90.0 / 1.6 | no |

| LIB | build median (s) | build min (s) | create median (s) |
| --- | --- | --- | --- |
| vsag | 15.229 | 15.036 | 15.196 |
| hipvs | 0.854 | 0.784 | 0.834 |

**Build speedup (median vsag / median hipvs): 17.83x** (DDL statement only: 18.21x)

## Recall@10 and single query latency

Exact ground truth, 100 queries x 1 rounds per setting, `SET ob_hnsw_ef_search`.

| LIB | ef_search | recall@10 | p50 (ms) | p95 (ms) | invalid result sets |
| --- | --- | --- | --- | --- | --- |
| vsag | 64 | 0.5790 | 0.35 | 0.38 | 0 |
| vsag | 128 | 0.7280 | 0.41 | 0.43 | 0 |
| vsag | 256 | 0.8660 | 0.56 | 0.57 | 0 |
| vsag | 512 | 0.9420 | 0.84 | 0.87 | 0 |
| hipvs | 64 | 0.8680 | 0.87 | 0.90 | 0 |
| hipvs | 128 | 0.9630 | 1.35 | 1.38 | 0 |
| hipvs | 256 | 0.9990 | 3.48 | 3.56 | 0 |
| hipvs | 512 | 1.0000 | 0.65 | 0.71 | 0 |

## GPU evidence

- rocm-smi during GPU builds: peak GPU use 100%, peak VRAM 1483 MiB (sampled every 0.2s, short builds may fall between samples)
- observer log, hipVS build lines:

```text
[2026-10-28 18:39:06.700987] algo=exact_knn rows=50000 knn_degree=128 graph_degree=64 knn_us=304770 optimize_us=93241 load_us=1650 host_threads=16 cost_us=400982 device=AMD Radeon Graphics
[2026-10-28 18:39:23.701089] algo=exact_knn rows=50000 knn_degree=128 graph_degree=64 knn_us=267111 optimize_us=91308 load_us=1634 host_threads=16 cost_us=363382 device=AMD Radeon Graphics
[2026-10-28 18:39:40.353726] algo=exact_knn rows=50000 knn_degree=128 graph_degree=64 knn_us=268411 optimize_us=90008 load_us=1632 host_threads=16 cost_us=362695 device=AMD Radeon Graphics
```
- CPU fallback warnings during GPU builds: 0
