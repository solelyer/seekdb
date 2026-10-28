# hipVS HNSW build benchmark

Generated: 2026-10-28 18:38:46 UTC

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

- rows: `10000`
- dim: `128`
- queries: `100`
- topk: `10`
- metric: `l2`
- m: `16`
- ef_construction: `200`
- ef_search: `64`
- runs: `5`
- rounds: `3`
- dataset: `/seekdb_workspace/datasets (truth gt_100x10.i32, agreement with numpy exact search 1.0000)`
- index DDL: `CREATE VECTOR INDEX idx_vec ON `bench_<lib>` (vec) WITH (DISTANCE=l2, TYPE=HNSW, LIB=<LIB>, M=16, EF_CONSTRUCTION=200, EF_SEARCH=64)`

## Separately recorded preparation

| step | seconds |
| --- | --- |
| dataset load / generation + exact truth (client) | 0.02 |
| base table import (no index) | 0.51 |
| copy into bench_vsag (no index) | 0.06 |
| copy into bench_hipvs (no index) | 0.06 |

## Build: CREATE VECTOR INDEX until the first query returns

Each run starts from a table with data and no vector index. `create` is the DDL statement, `first query` the first APPROXIMATE query afterwards; `build` = both.

| LIB | run | create (s) | first query (s) | build (s) | GPU path (observer log) | kNN / optimize / load (ms) | CPU fallback |
| --- | --- | --- | --- | --- | --- | --- | --- |
| vsag | 1 | 2.202 | 0.011 | 2.213 | - | - | n/a |
| vsag | 2 | 2.138 | 0.009 | 2.147 | - | - | n/a |
| vsag | 3 | 2.137 | 0.008 | 2.145 | - | - | n/a |
| vsag | 4 | 2.137 | 0.007 | 2.144 | - | - | n/a |
| vsag | 5 | 2.137 | 0.009 | 2.145 | - | - | n/a |
| hipvs | 1 | 0.626 | 0.046 | 0.672 | exact_knn | 329.8 / 17.7 / 0.7 | no |
| hipvs | 2 | 0.205 | 0.014 | 0.219 | exact_knn | 26.7 / 19.6 / 0.5 | no |
| hipvs | 3 | 0.236 | 0.013 | 0.249 | exact_knn | 25.8 / 18.4 / 0.4 | no |
| hipvs | 4 | 0.206 | 0.013 | 0.219 | exact_knn | 21.0 / 17.1 / 0.4 | no |
| hipvs | 5 | 0.208 | 0.013 | 0.222 | exact_knn | 20.2 / 16.2 / 0.4 | no |

| LIB | build median (s) | build min (s) | create median (s) |
| --- | --- | --- | --- |
| vsag | 2.145 | 2.144 | 2.137 |
| hipvs | 0.222 | 0.219 | 0.208 |

**Build speedup (median vsag / median hipvs): 9.69x** (DDL statement only: 10.25x)

## Recall@10 and single query latency

Exact ground truth, 100 queries x 3 rounds per setting, `SET ob_hnsw_ef_search`.

| LIB | ef_search | recall@10 | p50 (ms) | p95 (ms) | invalid result sets |
| --- | --- | --- | --- | --- | --- |
| vsag | 64 | 0.6400 | 0.30 | 0.32 | 0 |
| vsag | 128 | 0.8290 | 0.36 | 0.37 | 0 |
| vsag | 256 | 0.9540 | 0.47 | 0.48 | 0 |
| vsag | 512 | 0.9940 | 0.66 | 0.68 | 0 |
| hipvs | 64 | 0.9910 | 0.84 | 0.86 | 0 |
| hipvs | 128 | 1.0000 | 1.33 | 1.36 | 0 |
| hipvs | 256 | 1.0000 | 3.36 | 3.50 | 0 |
| hipvs | 512 | 1.0000 | 0.68 | 0.69 | 0 |

## GPU evidence

- rocm-smi during GPU builds: peak GPU use 99%, peak VRAM 1192 MiB (sampled every 0.2s, short builds may fall between samples)
- observer log, hipVS build lines:

```text
[2026-10-28 18:38:30.872448] algo=exact_knn rows=10000 knn_degree=128 graph_degree=64 knn_us=329759 optimize_us=17701 load_us=713 host_threads=16 cost_us=349762 device=AMD Radeon Graphics
[2026-10-28 18:38:34.244453] algo=exact_knn rows=10000 knn_degree=128 graph_degree=64 knn_us=26665 optimize_us=19596 load_us=467 host_threads=16 cost_us=48794 device=AMD Radeon Graphics
[2026-10-28 18:38:37.473273] algo=exact_knn rows=10000 knn_degree=128 graph_degree=64 knn_us=25774 optimize_us=18402 load_us=426 host_threads=16 cost_us=45876 device=AMD Radeon Graphics
[2026-10-28 18:38:40.646550] algo=exact_knn rows=10000 knn_degree=128 graph_degree=64 knn_us=20995 optimize_us=17106 load_us=422 host_threads=16 cost_us=39814 device=AMD Radeon Graphics
[2026-10-28 18:38:43.624889] algo=exact_knn rows=10000 knn_degree=128 graph_degree=64 knn_us=20200 optimize_us=16243 load_us=409 host_threads=16 cost_us=38247 device=AMD Radeon Graphics
```
- CPU fallback warnings during GPU builds: 0
