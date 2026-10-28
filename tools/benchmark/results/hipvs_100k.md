# hipVS HNSW build benchmark

Generated: 2026-10-28 08:59:17

## Environment

- server_version: `OceanBase 4.3.5.3 seekdb (r1.4.0.0) (Built 1-6cc4b91ae856856e4041c3144609f39bc5f2143d Oct 28 2026)`
- uname: `Linux 6.8.0-79-generic`
- host: `127.0.0.1`
- port: `2882`

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
- rounds: `3`
- data_dir: `generated (seed 20261028)`

## Build (CREATE VECTOR INDEX over 100000 rows)

Base table load (no index): **5.22s**

| LIB | build median (s) | build min (s) | build runs (s) | load (s) | GPU peak (%) | recall@10 | p50 (ms) | p95 (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vsag | 38.62 | 38.24 | 38.24, 38.62, 40.03 | 0.52 | 7.0 | 0.5090 | 0.36 | 0.45 |
| hipvs | 2.33 | 2.20 | 2.51, 2.20, 2.33 | 0.54 | 43.0 | 0.5740 | 0.89 | 0.93 |

Add the reproduction command, hardware facts and the GPU evidence snippet (rocm-smi tail) below before submitting.

