#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""End-to-end HNSW build benchmark for seekdb LIB=VSAG (CPU) vs LIB=HIPVS (AMD GPU).

The benchmark only talks SQL to a running seekdb instance and follows the
competition measurement rules:

  1. data preparation (generation or loading the shipped files) and the base
     table import are timed separately and never count as build time,
  2. every build starts from a table that holds the data but no vector index
     and runs `CREATE VECTOR INDEX`; the build ends when the index answers its
     first APPROXIMATE query, so work deferred to the first query is included,
  3. builds are repeated --runs times, interleaved between backends to spread
     drift evenly, raw times are kept and the speedup uses the medians,
  4. afterwards Recall@k and single query latency are measured for every
     ob_hnsw_ef_search value in --ef-search-list against exact ground truth,
     and every returned row set is checked (k rows, unique, valid ids),
  5. with --server-log the observer log of each build is scanned for the hipVS
     build line (algorithm, kNN / optimize / load time, device) and for CPU
     fallback warnings, and rocm-smi is sampled while the GPU builds run.

Usage:
  pip install numpy pymysql
  # competition dataset shipped with the AMD instance
  python3 tools/benchmark/hipvs_bench.py --port 2881 \
      --data-dir /seekdb_workspace/datasets --libs vsag,hipvs --runs 3 \
      --server-log /path/to/observer/log/seekdb.log \
      --output hipvs_bench.json --report hipvs_report.md
  # generated data for scaling runs (fixed seed)
  python3 tools/benchmark/hipvs_bench.py --rows 100000 --dim 128 ...

Required server side: seekdb built with -DOB_USE_HIPVS=ON for LIB=HIPVS.
"""

import argparse
import glob
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
import traceback

try:
    import numpy as np
except ImportError:  # pragma: no cover
    sys.exit("numpy is required: pip install numpy")

try:
    import pymysql
except ImportError:  # pragma: no cover
    sys.exit("pymysql is required: pip install pymysql")

METRIC_SQL = {
    "l2": ("l2", "l2_distance"),
    "cosine": ("cosine", "cosine_distance"),
}

# Observer log lines written by the hipVS bridge.
GPU_BUILD_RE = re.compile(r"hipvs (?:graph built before serialize|graph built|build finished)\((.*)\)")
FALLBACK_RE = re.compile(
    r"hipvs (?:graph build before serialize failed|graph build failed|exact knn graph build failed"
    r"|rebuild after a failed extend failed|extend failed|backend is not available)")


def log(msg):
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def vector_literal(vec):
    return "[" + ",".join("%.7g" % float(v) for v in vec) + "]"


def connect(args):
    return pymysql.connect(
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
        charset="utf8mb4",
        autocommit=True,
        connect_timeout=10,
        read_timeout=args.query_timeout,
    )


def execute(conn, sql, args=None):
    with conn.cursor() as cur:
        cur.execute(sql, args)
        return cur.fetchall()


def drop_table(conn, name):
    execute(conn, "DROP TABLE IF EXISTS `%s`" % name)


def bulk_insert(conn, table, vectors, batch):
    total = len(vectors)
    done = 0
    while done < total:
        chunk = min(batch, total - done)
        values = []
        params = []
        for i in range(done, done + chunk):
            values.append("(%s,%s)")
            params.append(i)
            params.append(vector_literal(vectors[i]))
        execute(conn, "INSERT INTO `%s` (id, vec) VALUES " % table + ",".join(values), params)
        done += chunk
        if done % (batch * 20) == 0 or done == total:
            log("  loaded %d/%d rows into %s" % (done, total, table))


def exact_truth(vectors, queries, topk, metric):
    """Exact top-k by brute force, chunked so that large sets fit in memory."""
    truth = []
    if metric == "l2":
        norms = (vectors * vectors).sum(axis=1)
    else:
        norms = np.linalg.norm(vectors, axis=1)
        norms[norms == 0.0] = 1.0
    for q in queries:
        if metric == "l2":
            dist = norms - 2.0 * vectors.dot(q)
        else:
            qn = float(np.linalg.norm(q)) or 1.0
            dist = 1.0 - vectors.dot(q) / (norms * qn)
        idx = np.argpartition(dist, topk - 1)[:topk]
        truth.append(set(int(i) for i in idx))
    return truth


def load_dataset(args):
    """Returns (vectors, queries, truth, prep_sec, source)."""
    start = time.perf_counter()
    if args.data_dir:
        vectors = np.fromfile(os.path.join(args.data_dir, "base.f32"),
                              dtype=np.float32).reshape(-1, args.dim)
        queries = np.fromfile(os.path.join(args.data_dir, "query.f32"),
                              dtype=np.float32).reshape(-1, args.dim)
        gt_files = sorted(glob.glob(os.path.join(args.data_dir, "gt_*.i32")))
        if not gt_files:
            raise SystemExit("no gt_*.i32 ground truth under %s" % args.data_dir)
        gt = np.fromfile(gt_files[0], dtype=np.int32)
        args.rows = int(vectors.shape[0])
        args.queries = int(queries.shape[0])
        args.topk = int(gt.size // args.queries)
        truth = [set(int(i) for i in row) for row in gt.reshape(args.queries, args.topk)]
        # The shipped truth must agree with an exact search of the same files.
        check = exact_truth(vectors, queries, args.topk, args.metric)
        agree = sum(len(a & b) for a, b in zip(check, truth)) / float(args.queries * args.topk)
        source = "%s (truth %s, agreement with numpy exact search %.4f)" % (
            args.data_dir, os.path.basename(gt_files[0]), agree)
    else:
        rng = np.random.default_rng(args.seed)
        vectors = rng.standard_normal((args.rows, args.dim), dtype=np.float32)
        queries = rng.standard_normal((args.queries, args.dim), dtype=np.float32)
        truth = exact_truth(vectors, queries, args.topk, args.metric)
        source = "generated standard normal, seed %d" % args.seed
    prep_sec = time.perf_counter() - start
    log("dataset: %s rows=%d dim=%d queries=%d topk=%d (%.2fs)"
        % (source, args.rows, args.dim, args.queries, args.topk, prep_sec))
    return vectors, queries, truth, prep_sec, source


class GpuSampler(threading.Thread):
    """Samples rocm-smi GPU busy percentage and VRAM while a build runs."""

    def __init__(self, interval):
        super().__init__(daemon=True)
        self.interval = interval
        self.busy = []
        self.vram = []
        self.last_raw = ""
        self._stop_event = threading.Event()

    def run(self):
        while not self._stop_event.is_set():
            try:
                out = subprocess.run(["rocm-smi", "--showuse", "--showmemuse", "--showmeminfo", "vram"],
                                     capture_output=True, text=True, timeout=5).stdout
            except Exception:  # pragma: no cover
                return
            self.last_raw = out
            for m in re.finditer(r"GPU use \(%\):\s*(\d+)", out):
                self.busy.append(int(m.group(1)))
            for m in re.finditer(r"VRAM Total Used Memory \(B\):\s*(\d+)", out):
                self.vram.append(int(m.group(1)))
            self._stop_event.wait(self.interval)

    def stop(self):
        self._stop_event.set()
        self.join(timeout=10)


def scan_server_log(path, offset):
    """GPU build lines and fallback warnings written after offset."""
    evidence = {"builds": [], "fallbacks": []}
    if not path or not os.path.exists(path):
        return evidence
    with open(path, "rb") as f:
        size = os.path.getsize(path)
        # the log may have rotated in between
        f.seek(offset if offset <= size else 0)
        for raw in f:
            line = raw.decode("utf-8", "replace")
            m = GPU_BUILD_RE.search(line)
            if m:
                fields = dict(kv.split("=", 1) for kv in m.group(1).split(", ") if "=" in kv)
                fields["ts"] = line[1:27]
                evidence["builds"].append(fields)
            elif FALLBACK_RE.search(line):
                evidence["fallbacks"].append(line.strip()[:400])
    return evidence


def log_offset(path):
    return os.path.getsize(path) if path and os.path.exists(path) else 0


def create_index_sql(table, lib, args):
    return ("CREATE VECTOR INDEX idx_vec ON `%s` (vec) WITH "
            "(DISTANCE=%s, TYPE=HNSW, LIB=%s, M=%d, EF_CONSTRUCTION=%d, EF_SEARCH=%d)"
            % (table, METRIC_SQL[args.metric][0], lib.upper(), args.m,
               args.ef_construction, args.ef_search))


def query_sql(table, args):
    return ("SELECT id FROM `%s` ORDER BY %s(vec, %%s) APPROXIMATE LIMIT %d"
            % (table, METRIC_SQL[args.metric][1], args.topk))


def build_once(conn, lib, args, probe):
    """One timed build: CREATE VECTOR INDEX, then the first query."""
    table = "bench_%s" % lib
    sampler = GpuSampler(args.gpu_sample_interval) if lib != "vsag" else None
    offset = log_offset(args.server_log)
    if sampler:
        sampler.start()
    try:
        start = time.perf_counter()
        execute(conn, create_index_sql(table, lib, args))
        created = time.perf_counter()
        rows = execute(conn, query_sql(table, args), (vector_literal(probe),))
        queryable = time.perf_counter()
    finally:
        if sampler:
            sampler.stop()
    if len(rows) != args.topk:
        raise RuntimeError("first query after the build returned %d rows" % len(rows))
    run = {
        "create_sec": created - start,
        "first_query_sec": queryable - created,
        "build_sec": queryable - start,
    }
    time.sleep(0.2)  # let the observer flush its log
    run["log"] = scan_server_log(args.server_log, offset)
    if sampler:
        run["gpu_busy_max_pct"] = max(sampler.busy) if sampler.busy else None
        run["vram_used_max_mb"] = max(sampler.vram) / 2**20 if sampler.vram else None
        run["gpu_samples"] = len(sampler.busy)
        run["rocm_smi_tail"] = sampler.last_raw[-800:]
    return run


def measure_queries(conn, lib, args, queries, truth):
    table = "bench_%s" % lib
    sql = query_sql(table, args)
    out = []
    for ef in args.ef_search_list:
        execute(conn, "SET ob_hnsw_ef_search = %d" % ef)
        for i in range(args.warmup):
            execute(conn, sql, (vector_literal(queries[i % len(queries)]),))
        latencies = []
        hits = 0
        invalid = 0
        for _ in range(args.rounds):
            for qi, q in enumerate(queries):
                start = time.perf_counter()
                rows = execute(conn, sql, (vector_literal(q),))
                latencies.append((time.perf_counter() - start) * 1000.0)
                ids = [int(r[0]) for r in rows]
                if len(ids) != args.topk or len(set(ids)) != len(ids) \
                        or any(x < 0 or x >= args.rows for x in ids):
                    invalid += 1
                hits += len(set(ids) & truth[qi])
        latencies.sort()
        out.append({
            "ef_search": ef,
            "recall_at_k": hits / float(args.rounds * len(queries) * args.topk),
            "invalid_results": invalid,
            "queries": len(latencies),
            "p50_ms": latencies[len(latencies) // 2],
            "p95_ms": latencies[max(0, int(len(latencies) * 0.95) - 1)],
            "mean_ms": statistics.fmean(latencies),
        })
        log("  [%s] ef_search=%d recall@%d=%.4f p50=%.2fms invalid=%d"
            % (lib, ef, args.topk, out[-1]["recall_at_k"], out[-1]["p50_ms"], invalid))
    execute(conn, "SET ob_hnsw_ef_search = %d" % args.ef_search)
    return out


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:  # pragma: no cover
        return ""


def collect_env(conn, args):
    env = {
        "host": args.host,
        "port": args.port,
        "uname": " ".join(os.uname()[i] for i in (0, 2)),
        "cpu": sh("lscpu | sed -n 's/^Model name: *//p' | head -1"),
        "cpu_quota": sh("cat /sys/fs/cgroup/cpu.max 2>/dev/null"),
        "nproc": sh("nproc"),
        "memory": sh("free -g | awk '/Mem:/ {print $2\" GiB\"}'"),
        "gpu": sh("rocm-smi --showproductname 2>/dev/null | grep -E 'Card Series|GFX Version' | "
                  "sed 's/^GPU\\[0\\][[:space:]]*: //' | tr -s ' ' | paste -sd ';'"),
        "vram": sh("rocm-smi --showmeminfo vram 2>/dev/null | grep 'Total Memory' | awk '{print $NF/2^30\" GiB\"}'"),
        "rocm": sh("cat /opt/rocm/.info/version 2>/dev/null"),
        "hipvs": sh("grep -h -E 'define CUVS_VERSION_(MAJOR|MINOR|PATCH)' "
                    "/opt/hipvs/include/cuvs/version_config.h 2>/dev/null | awk '{print $3}' | paste -sd ."),
    }
    rows = execute(conn, "SELECT @@version_comment")
    env["server_version"] = str(rows[0][0]) if rows else None
    return env


def median(values):
    return statistics.median(values) if values else None


def summarize(results, libs):
    summary = {}
    for lib in libs:
        runs = results[lib]["runs"]
        summary[lib] = {
            "create_median_sec": median([r["create_sec"] for r in runs]),
            "build_median_sec": median([r["build_sec"] for r in runs]),
            "build_min_sec": min(r["build_sec"] for r in runs) if runs else None,
        }
    if "vsag" in summary and "hipvs" in summary and summary["hipvs"]["build_median_sec"]:
        summary["speedup_build"] = summary["vsag"]["build_median_sec"] / summary["hipvs"]["build_median_sec"]
        summary["speedup_create"] = summary["vsag"]["create_median_sec"] / summary["hipvs"]["create_median_sec"]
    return summary


def write_report(path, env, args, prep, results, summary):
    k = args.topk
    lines = ["# hipVS HNSW build benchmark", "",
             "Generated: %s" % time.strftime("%Y-%m-%d %H:%M:%S %Z"), "", "## Environment", ""]
    for key in ("server_version", "cpu", "cpu_quota", "nproc", "memory", "gpu", "vram", "rocm",
                "hipvs", "uname", "host", "port"):
        lines.append("- %s: `%s`" % (key, env.get(key)))
    lines += ["", "## Configuration", ""]
    for key in ("rows", "dim", "queries", "topk", "metric", "m", "ef_construction", "ef_search",
                "runs", "rounds"):
        lines.append("- %s: `%s`" % (key, getattr(args, key)))
    lines.append("- dataset: `%s`" % prep["source"])
    lines.append("- index DDL: `%s`" % create_index_sql("bench_<lib>", "<lib>", args))
    lines += ["", "## Separately recorded preparation", "",
              "| step | seconds |", "| --- | --- |",
              "| dataset load / generation + exact truth (client) | %.2f |" % prep["dataset_sec"],
              "| base table import (no index) | %.2f |" % prep["base_load_sec"]]
    for lib in args.libs:
        lines.append("| copy into bench_%s (no index) | %.2f |" % (lib, results[lib]["copy_sec"]))
    lines += ["", "## Build: CREATE VECTOR INDEX until the first query returns", "",
              "Each run starts from a table with data and no vector index. `create` is the DDL "
              "statement, `first query` the first APPROXIMATE query afterwards; `build` = both.", "",
              "| LIB | run | create (s) | first query (s) | build (s) | GPU path (observer log) | "
              "kNN / optimize / load (ms) | CPU fallback |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for lib in args.libs:
        for i, r in enumerate(results[lib]["runs"]):
            builds = r["log"]["builds"]
            gpu_path = ", ".join(sorted(set(b.get("algo", "?") for b in builds))) if builds else "-"
            phases = "-"
            if builds:
                b = builds[0]
                phases = "%s / %s / %s" % tuple(
                    "%.1f" % (float(b.get(f, 0)) / 1000.0) for f in ("knn_us", "optimize_us", "load_us"))
            fallback = "yes (%d)" % len(r["log"]["fallbacks"]) if r["log"]["fallbacks"] else "no"
            lines.append("| %s | %d | %.3f | %.3f | %.3f | %s | %s | %s |"
                         % (lib, i + 1, r["create_sec"], r["first_query_sec"], r["build_sec"],
                            gpu_path, phases, fallback if lib != "vsag" else "n/a"))
    lines += ["", "| LIB | build median (s) | build min (s) | create median (s) |",
              "| --- | --- | --- | --- |"]
    for lib in args.libs:
        s = summary[lib]
        lines.append("| %s | %.3f | %.3f | %.3f |" % (lib, s["build_median_sec"], s["build_min_sec"],
                                                    s["create_median_sec"]))
    if "speedup_build" in summary:
        lines += ["", "**Build speedup (median vsag / median hipvs): %.2fx** "
                  "(DDL statement only: %.2fx)" % (summary["speedup_build"], summary["speedup_create"])]
    lines += ["", "## Recall@%d and single query latency" % k, "",
              "Exact ground truth, %d queries x %d rounds per setting, `SET ob_hnsw_ef_search`." %
              (args.queries, args.rounds), "",
              "| LIB | ef_search | recall@%d | p50 (ms) | p95 (ms) | invalid result sets |" % k,
              "| --- | --- | --- | --- | --- | --- |"]
    for lib in args.libs:
        for q in results[lib]["queries"]:
            lines.append("| %s | %d | %.4f | %.2f | %.2f | %d |"
                         % (lib, q["ef_search"], q["recall_at_k"], q["p50_ms"], q["p95_ms"],
                            q["invalid_results"]))
    gpu_runs = [r for lib in args.libs if lib != "vsag" for r in results[lib]["runs"]]
    if gpu_runs:
        lines += ["", "## GPU evidence", ""]
        busy = [r.get("gpu_busy_max_pct") for r in gpu_runs if r.get("gpu_busy_max_pct") is not None]
        vram = [r.get("vram_used_max_mb") for r in gpu_runs if r.get("vram_used_max_mb") is not None]
        if busy:
            lines.append("- rocm-smi during GPU builds: peak GPU use %d%%, peak VRAM %.0f MiB "
                         "(sampled every %.1fs, short builds may fall between samples)"
                         % (max(busy), max(vram) if vram else 0, args.gpu_sample_interval))
        builds = [b for r in gpu_runs for b in r["log"]["builds"]]
        if builds:
            lines += ["- observer log, hipVS build lines:", "", "```text"]
            for b in builds[:6]:
                lines.append("[%s] algo=%s rows=%s knn_degree=%s graph_degree=%s knn_us=%s "
                             "optimize_us=%s load_us=%s host_threads=%s cost_us=%s device=%s"
                             % (b.get("ts"), b.get("algo"), b.get("index->count()"), b.get("knn_degree"),
                                b.get("graph_degree"), b.get("knn_us"), b.get("optimize_us"),
                                b.get("load_us"), b.get("host_threads"), b.get("cost_us"),
                                b.get("capability().device_name")))
            lines.append("```")
        fallbacks = [f for r in gpu_runs for f in r["log"]["fallbacks"]]
        lines.append("- CPU fallback warnings during GPU builds: %d" % len(fallbacks))
        for f in fallbacks[:5]:
            lines.append("  - `%s`" % f)
        if not args.server_log:
            lines.append("- (no --server-log given, observer evidence not collected)")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log("wrote report %s" % path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default=os.environ.get("SEEKDB_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("SEEKDB_PORT", "2881")))
    parser.add_argument("--user", default=os.environ.get("SEEKDB_USER", "root"))
    parser.add_argument("--password", default=os.environ.get("SEEKDB_PASSWORD", ""))
    parser.add_argument("--database", default="hipvs_bench")
    parser.add_argument("--rows", type=int, default=100000)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--batch", type=int, default=500)
    parser.add_argument("--metric", choices=sorted(METRIC_SQL), default="l2")
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--ef-search", type=int, default=64)
    parser.add_argument("--ef-search-list", default="64,128,256,512",
                        help="ob_hnsw_ef_search values for the recall / latency sweep")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3, help="index build repetitions per LIB")
    parser.add_argument("--libs", default="vsag,hipvs")
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--data-dir", default="",
                        help="directory with base.f32, query.f32 and gt_*.i32 "
                             "(the competition dataset layout), overrides --rows/--queries/--topk")
    parser.add_argument("--server-log", default="",
                        help="observer seekdb.log, scanned for hipVS build and fallback lines")
    parser.add_argument("--gpu-sample-interval", type=float, default=0.2)
    parser.add_argument("--query-timeout", type=int, default=3600)
    parser.add_argument("--output", default="hipvs_bench.json")
    parser.add_argument("--report", default="hipvs_report.md")
    args = parser.parse_args()
    args.libs = [v.strip().lower() for v in args.libs.split(",") if v.strip() and v.strip() != "none"]
    args.ef_search_list = [int(v) for v in args.ef_search_list.split(",") if v.strip()]
    if args.runs < 3:
        log("warning: the competition rules ask for at least 3 runs per backend")
    return args


def main():
    args = parse_args()
    vectors, queries, truth, dataset_sec, source = load_dataset(args)
    conn = connect(args)
    results = {lib: {"runs": [], "queries": [], "copy_sec": None, "error": None} for lib in args.libs}
    try:
        execute(conn, "CREATE DATABASE IF NOT EXISTS `%s`" % args.database)
        execute(conn, "USE `%s`" % args.database)
        execute(conn, "SET ob_hnsw_ef_search = %d" % args.ef_search)
        env = collect_env(conn, args)
        log("loading the shared base table")
        drop_table(conn, "bench_base")
        execute(conn, "CREATE TABLE `bench_base` (id INT PRIMARY KEY, vec VECTOR(%d)) "
                      "ORGANIZATION = HEAP" % args.dim)
        start = time.perf_counter()
        bulk_insert(conn, "bench_base", vectors, args.batch)
        base_load_sec = time.perf_counter() - start
        for lib in args.libs:
            table = "bench_%s" % lib
            drop_table(conn, table)
            execute(conn, "CREATE TABLE `%s` (id INT PRIMARY KEY, vec VECTOR(%d)) "
                          "ORGANIZATION = HEAP" % (table, args.dim))
            start = time.perf_counter()
            execute(conn, "INSERT INTO `%s` SELECT id, vec FROM bench_base" % table)
            results[lib]["copy_sec"] = time.perf_counter() - start
            count = execute(conn, "SELECT COUNT(*) FROM `%s`" % table)[0][0]
            if count != args.rows:
                raise RuntimeError("%s holds %d rows, expected %d" % (table, count, args.rows))
        for run in range(args.runs):
            for lib in args.libs:
                if results[lib]["error"]:
                    continue
                try:
                    r = build_once(conn, lib, args, queries[0])
                    results[lib]["runs"].append(r)
                    log("  [%s] run %d: create %.3fs + first query %.3fs = %.3fs"
                        % (lib, run + 1, r["create_sec"], r["first_query_sec"], r["build_sec"]))
                    if run + 1 < args.runs:
                        execute(conn, "DROP INDEX idx_vec ON `bench_%s`" % lib)
                except Exception as exc:
                    results[lib]["error"] = "%s: %s" % (type(exc).__name__, exc)
                    log("  [%s] failed: %s\n%s" % (lib, results[lib]["error"], traceback.format_exc()))
        for lib in args.libs:
            if not results[lib]["error"]:
                results[lib]["queries"] = measure_queries(conn, lib, args, queries, truth)
        for lib in args.libs:
            drop_table(conn, "bench_%s" % lib)
        drop_table(conn, "bench_base")
    finally:
        conn.close()

    ok_libs = [lib for lib in args.libs if not results[lib]["error"]]
    summary = summarize(results, ok_libs)
    prep = {"source": source, "dataset_sec": dataset_sec, "base_load_sec": base_load_sec}
    payload = {"env": env, "config": {k: v for k, v in vars(args).items()}, "prep": prep,
               "results": results, "summary": summary}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    log("wrote %s" % args.output)
    args.libs = ok_libs
    write_report(args.report, env, args, prep, results, summary)
    if "speedup_build" in summary:
        log("build median vsag %.3fs hipvs %.3fs speedup %.2fx"
            % (summary["vsag"]["build_median_sec"], summary["hipvs"]["build_median_sec"],
               summary["speedup_build"]))
    return 1 if len(ok_libs) != len(results) else 0


if __name__ == "__main__":
    sys.exit(main())
