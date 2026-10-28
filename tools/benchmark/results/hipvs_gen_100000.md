# hipVS HNSW build benchmark

Generated: 2026-10-28 18:41:54 UTC

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

- rows: `100000`
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
| dataset load / generation + exact truth (client) | 0.72 |
| base table import (no index) | 4.82 |
| copy into bench_vsag (no index) | 0.42 |
| copy into bench_hipvs (no index) | 0.41 |

## Build: CREATE VECTOR INDEX until the first query returns

Each run starts from a table with data and no vector index. `create` is the DDL statement, `first query` the first APPROXIMATE query afterwards; `build` = both.

| LIB | run | create (s) | first query (s) | build (s) | GPU path (observer log) | kNN / optimize / load (ms) | CPU fallback |
| --- | --- | --- | --- | --- | --- | --- | --- |
| vsag | 1 | 38.068 | 0.060 | 38.129 | - | - | n/a |
| vsag | 2 | 38.832 | 0.059 | 38.891 | - | - | n/a |
| vsag | 3 | 38.992 | 0.060 | 39.052 | - | - | n/a |
| hipvs | 1 | 2.026 | 0.027 | 2.053 | exact_knn | 1006.3 / 216.6 / 3.1 | no |
| hipvs | 2 | 1.939 | 0.027 | 1.966 | exact_knn | 962.4 / 222.1 / 3.1 | no |
| hipvs | 3 | 1.931 | 0.024 | 1.955 | exact_knn | 963.7 / 215.0 / 3.1 | no |

| LIB | build median (s) | build min (s) | create median (s) |
| --- | --- | --- | --- |
| vsag | 38.891 | 38.129 | 38.832 |
| hipvs | 1.966 | 1.955 | 1.939 |

**Build speedup (median vsag / median hipvs): 19.78x** (DDL statement only: 20.02x)

## Recall@10 and single query latency

Exact ground truth, 100 queries x 1 rounds per setting, `SET ob_hnsw_ef_search`.

| LIB | ef_search | recall@10 | p50 (ms) | p95 (ms) | invalid result sets |
| --- | --- | --- | --- | --- | --- |
| vsag | 64 | 0.5090 | 0.35 | 0.38 | 0 |
| vsag | 128 | 0.6490 | 0.43 | 0.45 | 0 |
| vsag | 256 | 0.7840 | 0.61 | 0.63 | 0 |
| vsag | 512 | 0.8830 | 0.97 | 1.01 | 0 |
| hipvs | 64 | 0.7490 | 0.88 | 0.92 | 0 |
| hipvs | 128 | 0.9140 | 1.39 | 1.41 | 0 |
| hipvs | 256 | 0.9830 | 3.55 | 3.67 | 0 |
| hipvs | 512 | 0.9990 | 0.65 | 0.66 | 0 |

## GPU evidence

- rocm-smi during GPU builds: peak GPU use 100%, peak VRAM 1679 MiB (sampled every 0.2s, short builds may fall between samples)
- observer log, hipVS build lines:

```text
[2026-10-28 18:40:29.267258] algo=exact_knn rows=100000 knn_degree=128 graph_degree=64 knn_us=1006299 optimize_us=216643 load_us=3074 host_threads=16 cost_us=1226872 device=AMD Radeon Graphics
[2026-10-28 18:41:11.079571] algo=exact_knn rows=100000 knn_degree=128 graph_degree=64 knn_us=962428 optimize_us=222126 load_us=3074 host_threads=16 cost_us=1188969 device=AMD Radeon Graphics
[2026-10-28 18:41:52.865917] algo=exact_knn rows=100000 knn_degree=128 graph_degree=64 knn_us=963702 optimize_us=214960 load_us=3071 host_threads=16 cost_us=1183186 device=AMD Radeon Graphics
```
- CPU fallback warnings during GPU builds: 0
