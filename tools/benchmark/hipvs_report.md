# seekdb × AMD hipVS：HNSW 构建加速实验报告（模板）

> 用法：在 AMD GPU 机器上跑完 `hipvs_bench.py` 后，把
> `hipvs_report.md` 的表格与 `hipvs_bench.json` 的关键行贴进本模板，
> 补齐硬件信息和 GPU 证据即可提交。

## 1. 结论摘要

- 构建加速：LIB=HIPVS 相比 LIB=VSAG，`CREATE VECTOR INDEX` 中位数从 `__ s`
  降到 `__ s`，加速比 `__x`（数据量 `____` 行，维度 `__`，M=`__`，
  ef_construction=`__`）。
- 查询质量：Recall@10 从 `____`（VSAG）到 `____`（HIPVS），差异 `__`。
- 查询延迟：p50 `__ ms` vs `__ ms`。
- 适用边界：LIB=HIPVS 仅支持 `TYPE=HNSW`、`float32`、`L2/COSINE/INNER_PRODUCT`，
  不支持 SQ/BQ/SINDI；无 GPU 或未编译 hipVS 时 DDL 直接报错而不是静默回退。

## 2. 实现概述

- 编译开关：`cmake -DOB_USE_HIPVS=ON`，默认 OFF，普通构建零影响。
- 索引 opt-in：`WITH (..., LIB=HIPVS)`；缺省仍为 VSAG（HNSW）/ OB（IVF）。
- 后端桥接：`src/oblib/lib/vector/ob_hipvs_bridge.{h,cpp}` 把 seekdb 的
  HNSW 参数映射到 cuVS CAGRA（build / extend / search / serialize），
  `ob_vector_util.cpp` 按 `lib_` 分发，未命中注册表时原样走 vsag 指针。
- 增量 memtable 尚未生成 CAGRA 图，桥接层用主机端精确扫描兜底，
  保证 merge 前的新写入可见；搜索结果距离统一为“越小越好”（IP 取负）。
- 持久化：自定义自描述容器保存原始向量、vid、extra info 与 CAGRA 图，
  与 CPU 索引的序列化互不兼容、互不误读。

## 3. 复现步骤

```bash
# 1) 编译（ROCm 环境，hipVS 安装到 /opt/rocm 或传 -DOB_HIPVS_DIR=<prefix>）
cmake -S seekdb -B build_hipvs -DOB_USE_HIPVS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build_hipvs -j"$(nproc)"

# 2) 启动 seekdb（按团队既有部署脚本，监听 2881）

# 3) 依赖
pip install numpy pymysql

# 4) 一轮完整对比：无索引基线 / CPU / GPU
python3 tools/benchmark/hipvs_bench.py \
  --host 127.0.0.1 --port 2881 --user root \
  --rows 100000 --dim 128 --queries 100 --topk 10 \
  --libs none,vsag,hipvs --runs 3 \
  --output hipvs_bench.json --report hipvs_report.md
```

要求：`--runs >= 3`，报告取中位数；`--libs` 顺序固定；
三次运行之间不要有其他压测任务；同一份数据（脚本内固定随机种子）。

## 4. 环境

| 项目 | 值 |
| --- | --- |
| CPU | `____` |
| 内存 | `____` |
| GPU | `____`（`rocm-smi --showproductname`） |
| ROCm / hipVS 版本 | `____` |
| seekdb 版本 / commit | `____` |
| 编译选项 | `-DOB_USE_HIPVS=ON`，`CMAKE_BUILD_TYPE=Release` |

## 5. 构建耗时

（粘贴 `hipvs_report.md` 的 Build 表）

| LIB | build median (s) | build runs (s) | GPU peak (%) | recall@10 | p50 (ms) | p95 (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| none | - | - | - | - | - | - |
| vsag | | | - | | | |
| hipvs | | | | | | |

说明：`CREATE VECTOR INDEX idx_vec ON bench_<lib>(vec) WITH (DISTANCE=..., TYPE=HNSW, LIB=...)`
对已有数据做全量回填，计时为 SQL 端到端墙钟时间（含 DDL 调度与落盘）。

## 6. Recall 与查询延迟

- Ground truth：同一批向量在客户端用 numpy 精确计算（与 SQL 侧同精度浮点）。
- 查询 SQL：`SELECT id FROM t ORDER BY l2_distance(vec, '[...]') APPROXIMATE LIMIT 10`。
- 结果：

| LIB | recall@10 | p50 (ms) | p95 (ms) | 查询样本数 |
| --- | --- | --- | --- | --- |
| vsag | | | | |
| hipvs | | | | |

## 7. GPU 证据

1. 构建期间 `rocm-smi --showuse --showmemuse` 采样（脚本自动采集最高 GPU 利用率）：

```text
（粘贴 hipvs_bench.json 中 results[hipvs].gpu.raw_tail）
```

2. seekdb 日志（`grep -E "hipvs (index created|build finished|extend finished)" observer.log`）：

```text
（粘贴若干行，包含 device_name）
```

## 8. 局限与后续工作

- 当前只映射 dense float32 HNSW；SQ/BQ/IVF/SINDI 继续走 CPU 后端。
- 增量 memtable 走主机端扫描，图形化构建发生在快照回填；
  后续可在 dump 时机直接调用 cuVS build 减少一次全量拷贝。
- 序列化格式为 hipVS 私有；跨后端迁移需要重建索引。
- 多卡：当前固定 device 0，后续可按 tablet 亲和分配。
