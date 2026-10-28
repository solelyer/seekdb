# hipVS HNSW build benchmark

Generated: 2026-10-28 18:47:13 UTC

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

- rows: `200000`
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
| dataset load / generation + exact truth (client) | 1.15 |
| base table import (no index) | 9.38 |
| copy into bench_vsag (no index) | 0.92 |
| copy into bench_hipvs (no index) | 0.91 |

## Build: CREATE VECTOR INDEX until the first query returns

Each run starts from a table with data and no vector index. `create` is the DDL statement, `first query` the first APPROXIMATE query afterwards; `build` = both.

| LIB | run | create (s) | first query (s) | build (s) | GPU path (observer log) | kNN / optimize / load (ms) | CPU fallback |
| --- | --- | --- | --- | --- | --- | --- | --- |
| vsag | 1 | 95.513 | 0.122 | 95.636 | - | - | n/a |
| vsag | 2 | 99.042 | 0.117 | 99.159 | - | - | n/a |
| vsag | 3 | 96.921 | 0.125 | 97.046 | - | - | n/a |
| hipvs | 1 | 3.556 | 0.066 | 3.622 | ivf_flat_knn | 1166.8 / 567.9 / 6.0 | no |
| hipvs | 2 | 3.344 | 0.043 | 3.387 | ivf_flat_knn | 958.4 / 533.9 / 6.0 | no |
| hipvs | 3 | 3.352 | 0.039 | 3.391 | ivf_flat_knn | 956.4 / 534.5 / 6.0 | no |

| LIB | build median (s) | build min (s) | create median (s) |
| --- | --- | --- | --- |
| vsag | 97.046 | 95.636 | 96.921 |
| hipvs | 3.391 | 3.387 | 3.352 |

**Build speedup (median vsag / median hipvs): 28.61x** (DDL statement only: 28.91x)

## Recall@10 and single query latency

Exact ground truth, 100 queries x 1 rounds per setting, `SET ob_hnsw_ef_search`.

| LIB | ef_search | recall@10 | p50 (ms) | p95 (ms) | invalid result sets |
| --- | --- | --- | --- | --- | --- |
| vsag | 64 | 0.3660 | 0.78 | 0.95 | 0 |
| vsag | 128 | 0.5290 | 0.72 | 0.85 | 0 |
| vsag | 256 | 0.6880 | 0.93 | 1.08 | 0 |
| vsag | 512 | 0.8070 | 1.33 | 1.47 | 0 |
| hipvs | 64 | 0.6050 | 0.98 | 1.02 | 0 |
| hipvs | 128 | 0.8020 | 1.50 | 1.54 | 0 |
| hipvs | 256 | 0.9340 | 3.82 | 3.95 | 0 |
| hipvs | 512 | 0.9880 | 0.69 | 0.73 | 0 |

## GPU evidence

- rocm-smi during GPU builds: peak GPU use 100%, peak VRAM 1917 MiB (sampled every 0.2s, short builds may fall between samples)
- observer log, hipVS build lines:

```text
[2026-10-28 18:43:46.846195] algo=ivf_flat_knn rows=200000 knn_degree=128 graph_degree=64 knn_us=1166813 optimize_us=567924 load_us=5960 host_threads=16 cost_us=1742399 device=AMD Radeon Graphics
[2026-10-28 18:45:30.434864] algo=ivf_flat_knn rows=200000 knn_degree=128 graph_degree=64 knn_us=958447 optimize_us=533913 load_us=5962 host_threads=16 cost_us=1500536 device=AMD Radeon Graphics
[2026-10-28 18:47:11.618253] algo=ivf_flat_knn rows=200000 knn_degree=128 graph_degree=64 knn_us=956378 optimize_us=534498 load_us=5962 host_threads=16 cost_us=1498319 device=AMD Radeon Graphics
```
- CPU fallback warnings during GPU builds: 0
