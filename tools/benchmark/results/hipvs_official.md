# hipVS HNSW build benchmark

Generated: 2026-10-28 08:56:19

## Environment

- server_version: `OceanBase 4.3.5.3 seekdb (r1.4.0.0) (Built 1-6cc4b91ae856856e4041c3144609f39bc5f2143d Oct 28 2026)`
- uname: `Linux 6.8.0-79-generic`
- host: `127.0.0.1`
- port: `2882`

## Configuration

- rows: `10000`
- dim: `128`
- queries: `100`
- topk: `10`
- metric: `l2`
- m: `16`
- ef_construction: `200`
- ef_search: `64`
- runs: `3`
- rounds: `3`
- data_dir: `/seekdb_workspace/datasets`

## Build (CREATE VECTOR INDEX over 10000 rows)

Base table load (no index): **0.52s**

| LIB | build median (s) | build min (s) | build runs (s) | load (s) | GPU peak (%) | recall@10 | p50 (ms) | p95 (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vsag | 2.32 | 2.32 | 2.45, 2.32, 2.32 | 0.05 | 0.0 | 0.6400 | 0.30 | 0.31 |
| hipvs | 1.23 | 1.16 | 1.83, 1.23, 1.16 | 0.05 | 22.0 | 0.9710 | 0.85 | 0.88 |

Add the reproduction command, hardware facts and the GPU evidence snippet (rocm-smi tail) below before submitting.

