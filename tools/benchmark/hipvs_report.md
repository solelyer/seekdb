# seekdb × AMD hipVS：HNSW 索引构建加速实验报告

## 1. 结论

同一 seekdb 二进制、同一批向量、同一组索引参数（`DISTANCE=l2, TYPE=HNSW, M=16,
EF_CONSTRUCTION=200, EF_SEARCH=64`）下，`CREATE VECTOR INDEX` 到首条查询返回的端到端构建耗时：

| 数据量 / 维度 | CPU 构建耗时（LIB=VSAG，中位数） | hipVS 构建耗时（LIB=HIPVS，中位数） | 构建加速比 | CPU / GPU Recall@10（ef_search=64） | GPU 构图路径 |
| --- | --- | --- | --- | --- | --- |
| 10,000 / 128（大赛数据集） | 2.145 s | 0.222 s | **9.69x** | 0.640 / 0.991 | exact_knn |
| 50,000 / 128（生成数据） | 15.229 s | 0.854 s | **17.83x** | 0.579 / 0.868 | exact_knn |
| 100,000 / 128（生成数据） | 38.891 s | 1.966 s | **19.78x** | 0.509 / 0.749 | exact_knn |
| 200,000 / 128（生成数据） | 97.046 s | 3.391 s | **28.61x** | 0.366 / 0.605 | ivf_flat_knn |

- 每组 CPU / GPU 交替各构建 3 次（10k 为 5 次），原始耗时见第 5 节，加速比按两组中位数计算。
- 所有规模下 GPU 索引的 Recall@10 都高于 CPU 索引；同一 ef_search 下 GPU 更高，
  ef_search=512 时 GPU 为 0.988~1.000，CPU 为 0.807~0.994（第 6 节）。
- 每次 GPU 构建在 observer 日志中都有 hipVS 构建记录（算法、kNN / 优化 / 装载耗时、设备名），
  CPU 回退告警为 0；rocm-smi 采样到 GPU 利用率 99%~100%（第 7 节）。
- 原有 CPU 路径（LIB=VSAG）行为不变：vector_index、fts_index、global_index、ddl、
  table_redefinition、storage 六个 mysqltest 套件在改动前后逐用例结果一致（第 8 节）。
- 是否“达到大赛公布的预期”需对照统一评测配置；题面目前未公布目标倍数与质量门槛。

## 2. 实现

### 2.1 接入位置

| 位置 | 作用 |
| --- | --- |
| `src/oblib/lib/vector/ob_hipvs_bridge.{h,cpp}` | hipVS（ROCm cuVS 1.0 C API）桥接：构建、增量、搜索、按 id 计算距离、快照序列化 |
| `src/oblib/lib/vector/ob_vector_util.cpp` | `obvectorutil` 按 `lib_` 分发到 hipVS 或 vsag，vsag 指针原样透传 |
| `src/observer/vector_index/ob_vector_index_util.cpp` | DDL 解析 `LIB=HIPVS`（别名 `LIB=CUVS`），只允许 `TYPE=HNSW` float32，拒绝量化 / 混合 / 稀疏；无可用设备时直接报 `OB_NOT_SUPPORTED`，不静默回退 |
| `src/storage/ddl/ob_ddl_pipeline.cpp` | 快照写完后通过当前登记的 adapter 重置快照（见 2.4） |
| `CMakeLists.txt`、`src/oblib/lib/CMakeLists.txt` | `-DOB_USE_HIPVS=ON` 才编译 hipVS 后端，默认构建不受影响 |
| `tools/benchmark/hipvs_bench.py` | 复现用的端到端基准脚本 |

### 2.2 构建链路

`CREATE VECTOR INDEX` 的回填阶段只把行追加到主机端缓冲区（向量、vid、extra info），
快照封存（serialize）时一次性在 GPU 上构图：

1. **kNN 图（GPU）**：数据集一次性拷到显存后求每行的 `intermediate_graph_degree` 个近邻。
   - `rows² × dim ≤ 131072² × 128`：cuVS 暴力检索，得到**精确** kNN 图；
   - 更大（≤ 4M 行）：IVF-Flat，`sqrt(rows)` 个簇，每行探测 32 个簇；
   - 上述路径失败或更大规模：沿用 cuVS 自带的 IVF-PQ 构图（主机端分批）。
2. **图优化（主机，最多 16 线程）**：按 cuVS `cagra::optimize` 的规则做基于名次的绕行剪枝，
   保留 `graph_degree` 条边，再用反向边替换后一半，保证中心节点可达。
3. **装载**：`cuvsCagraIndexFromArgs` 把图和数据集交给 cuVS，得到可直接搜索的 CAGRA 索引。
4. **持久化**：向量、vid、extra info 与图一起写入快照表；重启或其他节点加载时直接重建 CAGRA，不再构图。

参数映射：`graph_degree = clamp(4 × M, 32, 128)`（CAGRA 是单层图，与 HNSW 同为 M 时召回明显偏低），
`intermediate_graph_degree = max(2 × graph_degree, min(EF_CONSTRUCTION, 128))`；
搜索 `itopk = max(窗口, 2 × ef_search)`，`search_width` 随 ef_search 取 1 / 2 / 4，
`itopk > 512` 时改用 MULTI_CTA 内核。

### 2.3 为什么不用 cuVS 自带的构图

gfx1100（RDNA3）上 cuVS 的 IVF-PQ 构图受内核启动次数限制：10k × 128 的一次构建约有 7,800 次
`hipLaunchKernel` 和 4,500 次 `hipFree`（PQ 码本的 k-means），耗时约 0.8 s，GPU 大部分时间空闲；
NN-Descent、迭代搜索构图同样在 0.8~1.4 s。精确暴力检索在 10k 上只要约 20 ms，且得到的是真近邻，
剪枝后召回反而更高。独立测试（同参数、同机器）：

| 规模 | IVF-PQ 构图（cuVS 默认） | 精确 kNN + 主机优化 | IVF-Flat kNN + 主机优化 |
| --- | --- | --- | --- |
| 10k | 0.83 s，recall@10 0.97 | 0.04 s，0.99 | - |
| 100k | 1.50 s，0.58 | 1.23 s，0.78 | 0.63 s，0.74 |
| 200k | 2.17 s，0.43 | 4.29 s，0.65 | 1.78 s，0.60 |
| 500k | 4.27 s，0.24 | - | 9.72 s，0.43 |

（recall 为构图后 CAGRA 搜索 itopk=128、search_width=2 的 Recall@10。）

### 2.4 DDL 链路上的公共改动

构图降到几十毫秒后，10k 的端到端时间主要是 DDL 框架开销。以下改动对 CPU、GPU 两条路径同样生效：

| 提交 | 问题 | 效果（10k） |
| --- | --- | --- |
| `fix(ddl): only sleep between failed checksum report attempts` | 校验和上报成功后仍固定 `sleep 100ms` | 每次建索引 -100 ms |
| `perf(storage): wake idle dag threads when an independent dag finishes` | DAG 结束时不唤醒空闲线程，线程池回收最多再等 50 ms | -46 ms |
| `perf(ddl): poll DDL completion with a growing interval` | 客户端每 100 ms 轮询一次 DDL 结果 | 平均 -50 ms，长 DDL 仍按 100 ms 轮询 |

同一份 10k 数据：起点提交 `f348b4546` 上 LIB=VSAG 2.32 s、LIB=HIPVS 1.31 s；换成精确 kNN 构图、
尚未做这三项改动时 LIB=HIPVS 0.41 s；全部改动后 LIB=VSAG 2.145 s、LIB=HIPVS 0.222 s。
表 1 用的是同一二进制下两个后端的对比。

### 2.5 修复的正确性问题

| 提交 | 现象 | 原因 |
| --- | --- | --- |
| `fix(ddl): renew the vector snapshot through the registered adapter` | 200k 建索引后约 2/10 的概率查询全部返回 0 行，直到重建 | 向量索引调度器约每秒合并一次 partial adapter；若恰在快照写入期间合并，complete adapter 复制了最终快照前缀并共享快照内存，DDL 再经旧的 partial adapter 重置快照，查询匹配到前缀而跳过加载。VSAG 窗口只有快照写入时间，HIPVS 把构图放在这个窗口里所以更易触发。修复后 20/20 正常 |
| `fix(vector): load hipVS snapshots on the large-stack GPU thread` | 保存了 200k LIB=HIPVS 索引后重启，observer 在后台加载快照时退出 | `cuvsCagraIndexFromArgs` 在默认栈很小的 OB 线程上执行，与构建路径同样的栈溢出 |
| `fix(vector): use the multi-CTA CAGRA search past 512 candidates` | `ob_hnsw_ef_search > 256` 时每条查询报错 | AUTO 仍为单查询选单 CTA 内核，`itopk_size cannot be larger than 512` |

### 2.6 GPU 路径识别与回退

- 每次构图后输出 `hipvs graph built before serialize(... algo=exact_knn|ivf_flat_knn|ivf_pq,
  knn_us, optimize_us, load_us, host_threads, cost_us, device_name)`。
- 构图失败时输出 `hipvs graph build ... failed, keeping the host side scan` 等 WARN，快照中记录
  `has_graph=0`；`hipvs_bench.py --server-log` 会把这类告警统计为 CPU 回退并写进报告。
- 无设备或未编译 hipVS 时 `LIB=HIPVS` 的 DDL 直接报错，不会建成 CPU 索引冒充 GPU 结果。

## 3. 复现步骤

```bash
# 1) 编译（AMD 实例：ROCm 7.2.4，hipVS 位于 /opt/hipvs）
./build.sh release -DOB_USE_HIPVS=ON -DOB_HIPVS_DIR=/opt/hipvs --make -j64

# 2) 启动（与基准使用的参数一致）
export LD_LIBRARY_PATH=/opt/hipvs/lib:/opt/rocm/lib
build_release/src/observer/seekdb --base-dir=/workspace/run/obs_exact --port=2884 --nodaemon \
  --parameter memory_budget=16G --parameter cpu_count=64 --parameter datafile_size=20G \
  --parameter datafile_maxsize=40G --parameter log_disk_size=10G --log-level INFO &

# 3) 依赖
pip install numpy pymysql

# 4) 大赛数据集（base.f32 / query.f32 / gt_100x10.i32）
python3 tools/benchmark/hipvs_bench.py --port 2884 --data-dir /seekdb_workspace/datasets \
  --libs vsag,hipvs --runs 5 --rounds 3 \
  --server-log /workspace/run/obs_exact/log/seekdb.log \
  --output hipvs_official_10k.json --report hipvs_official_10k.md

# 5) 规模扩展（固定种子 20261028 的标准正态数据，ground truth 由 numpy 精确计算）
for n in 50000 100000 200000; do
  python3 tools/benchmark/hipvs_bench.py --port 2884 --rows $n --dim 128 \
    --libs vsag,hipvs --runs 3 --ef-search-list 64,128,256,512 \
    --server-log /workspace/run/obs_exact/log/seekdb.log \
    --output hipvs_gen_$n.json --report hipvs_gen_$n.md
done
```

计时口径（脚本内实现）：

- 起点：表已有数据、索引不存在；终点：`CREATE VECTOR INDEX` 返回后第一条 APPROXIMATE 查询返回 10 行。
  延迟到首次查询的工作计入构建时间；两组均重新构建，不复用索引或构图结果。
- 数据生成 / 文件读取与 ground truth、基表导入、复制到各后端的表分别计时，不计入构建时间。
- 两个后端交替构建，降低时间漂移的影响；保留每次原始耗时，取中位数计算加速比。
- 构建结束后对每个 `ob_hnsw_ef_search` 计算 Recall@10，并检查每个结果集：10 行、无重复、id 合法。
- 大赛数据集的 ground truth 与 numpy 精确检索逐条比对，一致率 1.0000。

## 4. 环境

| 项目 | 值 |
| --- | --- |
| CPU | 2 × AMD EPYC 9334 32-Core（128 线程），容器 CPU 配额 16 核（`cpu.max = 1600000 100000`） |
| 内存 | 503 GiB |
| GPU | AMD Radeon（gfx1100 / RDNA3，48 GiB 显存，rocm-smi Card Model 0x744b），单卡 |
| ROCm / hipVS | ROCm 7.2.4 / hipVS（cuVS）1.0.0，`/opt/hipvs` |
| seekdb | 1.4.0.0，`contest/hipvs-exact-knn` 分支提交 `37fdb9148`，RelWithDebInfo，`-DOB_USE_HIPVS=ON` |
| observer 参数 | `memory_budget=16G, cpu_count=64`，单机单副本 |

## 5. 构建耗时原始数据

`create` 为 DDL 语句耗时，`first query` 为随后第一条查询，`build = create + first query`。
GPU 列的 kNN / 优化 / 装载时间取自 observer 日志。

### 5.1 10,000 × 128（大赛数据集）

| LIB | 第 1 次 | 第 2 次 | 第 3 次 | 第 4 次 | 第 5 次 | 中位数 |
| --- | --- | --- | --- | --- | --- | --- |
| vsag | 2.213 | 2.147 | 2.145 | 2.144 | 2.145 | **2.145** |
| hipvs | 0.672 | 0.219 | 0.249 | 0.219 | 0.222 | **0.222** |
| hipvs kNN / 优化 / 装载 (ms) | 329.8 / 17.7 / 0.7 | 26.7 / 19.6 / 0.5 | 25.8 / 18.4 / 0.4 | 21.0 / 17.1 / 0.4 | 20.2 / 16.2 / 0.4 | |

hipVS 第 1 次是 observer 启动后的首次 GPU 构建，包含 HIP 代码对象加载（约 0.3 s），之后稳定在 0.22 s。
单独计时：数据文件读取 0.02 s，基表导入 0.51 s，复制到 bench 表 0.06 s。

### 5.2 规模扩展（生成数据）

| 规模 | LIB | 第 1 次 | 第 2 次 | 第 3 次 | 中位数 | 加速比 |
| --- | --- | --- | --- | --- | --- | --- |
| 50k | vsag | 15.036 | 15.325 | 15.229 | 15.229 | |
| 50k | hipvs | 0.867 | 0.784 | 0.854 | 0.854 | 17.83x |
| 100k | vsag | 38.129 | 38.891 | 39.052 | 38.891 | |
| 100k | hipvs | 2.053 | 1.966 | 1.955 | 1.966 | 19.78x |
| 200k | vsag | 95.636 | 99.159 | 97.046 | 97.046 | |
| 200k | hipvs | 3.622 | 3.387 | 3.391 | 3.391 | 28.61x |

hipVS kNN / 优化耗时（ms，第 2~3 次）：50k 约 268 / 91，100k 约 963 / 218，
200k（IVF-Flat）约 957 / 534。基表导入分别为 2.47 s、4.82 s、9.38 s，不计入构建。

## 6. Recall@10 与单条查询延迟

精确 ground truth，每个设置 100 条查询（10k 为 100 条 × 3 轮）。**单条查询延迟不是本题的构建指标，仅供参考。**

| 规模 | ef_search | VSAG recall | hipVS recall | VSAG p50 (ms) | hipVS p50 (ms) |
| --- | --- | --- | --- | --- | --- |
| 10k | 64 | 0.640 | 0.991 | 0.30 | 0.84 |
| 10k | 128 | 0.829 | 1.000 | 0.36 | 1.33 |
| 10k | 256 | 0.954 | 1.000 | 0.47 | 3.36 |
| 10k | 512 | 0.994 | 1.000 | 0.66 | 0.68 |
| 50k | 64 | 0.579 | 0.868 | 0.35 | 0.87 |
| 50k | 512 | 0.942 | 1.000 | 0.84 | 0.65 |
| 100k | 64 | 0.509 | 0.749 | 0.35 | 0.88 |
| 100k | 512 | 0.883 | 0.999 | 0.97 | 0.65 |
| 200k | 64 | 0.366 | 0.605 | 0.78 | 0.98 |
| 200k | 512 | 0.807 | 0.988 | 1.33 | 0.69 |

完整的 64 / 128 / 256 / 512 四档数据见 `results/hipvs_*.md`。所有结果集均为 10 行、无重复、id 合法（invalid = 0）。
生成数据是 128 维标准正态分布，本身难以检索，两种索引在大规模下的低 ef_search 召回都偏低；
ef_search=512 时 hipVS 使用 MULTI_CTA 内核，延迟反而低于 256。

## 7. GPU 路径证据

1. observer 日志（10k，5 次构建）：

```text
[2026-10-28 18:38:30.872448] algo=exact_knn rows=10000 knn_degree=128 graph_degree=64 knn_us=329759 optimize_us=17701 load_us=713 host_threads=16 cost_us=349762 device=AMD Radeon Graphics
[2026-10-28 18:38:34.244453] algo=exact_knn rows=10000 knn_degree=128 graph_degree=64 knn_us=26665 optimize_us=19596 load_us=467 host_threads=16 cost_us=48794 device=AMD Radeon Graphics
[2026-10-28 18:38:37.473273] algo=exact_knn rows=10000 knn_degree=128 graph_degree=64 knn_us=25774 optimize_us=18402 load_us=426 host_threads=16 cost_us=45876 device=AMD Radeon Graphics
[2026-10-28 18:38:40.646550] algo=exact_knn rows=10000 knn_degree=128 graph_degree=64 knn_us=20995 optimize_us=17106 load_us=422 host_threads=16 cost_us=39814 device=AMD Radeon Graphics
[2026-10-28 18:38:43.624889] algo=exact_knn rows=10000 knn_degree=128 graph_degree=64 knn_us=20200 optimize_us=16243 load_us=409 host_threads=16 cost_us=38247 device=AMD Radeon Graphics
```

   200k 的构建行为 `algo=ivf_flat_knn`。第 5 节全部 14 次 GPU 构建的 CPU 回退告警为 0。
2. rocm-smi（每 0.2 s 采样）：10k 构建期间 GPU 利用率峰值 99%（构建只有约 40 ms，5 次中有 3 次采到），
   50k~200k 每次均为 100%；显存峰值 1.2~1.9 GiB。
3. 内核级分析（独立程序 + rocprofv3，10k）：暴力 kNN 的 GEMM、select_k 与 CAGRA 搜索内核均在 gfx1100 上执行；
   对照的 IVF-PQ 构图两次构建共 15,637 次内核启动，其中 `sum_rows_by_key` 等 k-means 内核 273 ms，HIP API 调用共 1.78 s。

## 8. CPU 路径、正确性与稳定性

- **CPU 路径**：在同一数据目录上分别用改动前（`f348b4546`）与改动后的二进制重启并运行 mysqltest，
  逐用例结果一致：vector_index 10 / 25、fts_index 9 / 15、global_index 5 / 7、table_redefinition 2 / 2、
  storage 3 / 3、ddl 0 / 1 通过，失败用例两边完全相同：多数为 `-4019 Size overflow`，3 个需要 `admin` 账户，
  4 个结果文件不一致（`dump_ddl_mem`、`global_index_lookup_1`、`all_virtual_vector_index_info`、
  `vector_ivf_index_with_filter`）。`Size overflow` 是本实例的环境问题：observer 运行一段时间后 keep alive
  日志序列化失败（`min_start_status_`，32 字节缓冲），走 PX / 分区表的查询报同样错误，未改动的旧 observer（端口 2882）
  上的普通分区表查询也会出现，与本次修改无关。
- **重复构建**：同一张 10k 表连续 40 次 `CREATE` / 查询 / `DROP INDEX`，每次 20 条查询的 Recall@10 均为 0.995，
  显存在 1.3~1.4 GiB 间波动无增长，observer RSS 在 2.1~2.6 GiB 间波动（VSAG 同样波动）。
- **建后 DML**：插入与查询向量相同的新行后，它成为第一近邻；删除的真近邻不再返回；更新后的行能被查到。
- **首查竞争**：200k 表连续 20 次建索引 + 首查，修复前 2 / 10 次首查为空，修复后 0 / 20。
- **重启**：保存了 200k LIB=HIPVS 索引的 observer 重启后正常加载并查询。

## 9. 局限

- 仅映射 dense float32 HNSW；SQ / BQ / IVF / 稀疏索引继续走 CPU 后端。
- 增量写入（delta buffer）在主机端线性扫描，直到下一次快照合并时在 GPU 上重建图。
- 固定使用 device 0；4M 行以上回到 IVF-PQ 构图（主机端分批）。
- 主机端图优化使用最多 16 个线程，CPU 基线（VSAG）的 DDL 构建为单线程，这部分 CPU 开销计入了 GPU 组的端到端时间。
- 序列化格式为 hipVS 私有，与 VSAG 索引不能互相加载，切换后端需要重建索引。
