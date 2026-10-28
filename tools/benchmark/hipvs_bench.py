#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""End-to-end HNSW benchmark for seekdb LIB=VSAG (CPU) vs LIB=HIPVS (AMD GPU).

The benchmark only talks SQL to a running seekdb instance:

  1. load a base table without any vector index (shared data set),
  2. copy it into one table per backend and time `CREATE VECTOR INDEX`, which
     backfills the HNSW graph from the existing rows (the build we care about),
  3. measure APPROXIMATE top-k latency and Recall@k against a numpy ground
     truth computed from the same vectors,
  4. sample rocm-smi while the GPU build runs as hardware evidence.

Usage:
  pip install numpy pymysql
  python3 tools/benchmark/hipvs_bench.py --rows 100000 --dim 128 \
      --libs none,vsag,hipvs --runs 3 --output hipvs_bench.json \
      --report hipvs_report.md

Required server side: seekdb built with -DOB_USE_HIPVS=ON for LIB=HIPVS.
"""

import argparse
import json
import os
import re
import statistics
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


def table_exists(conn, name):
    rows = execute(conn, "SHOW TABLES LIKE %s", (name,))
    return len(rows) > 0


def drop_table(conn, name):
    execute(conn, "DROP TABLE IF EXISTS `%s`" % name)


def bulk_insert(conn, table, ids, vectors, tags, batch):
    total = len(ids)
    done = 0
    while done < total:
        chunk = min(batch, total - done)
        values = []
        params = []
        for i in range(done, done + chunk):
            values.append("(%s,%s,%s)")
            params.append(ids[i])
            params.append(vector_literal(vectors[i]))
            params.append(tags[i])
        sql = "INSERT INTO `%s` (id, vec, tag) VALUES " % table + ",".join(values)
        execute(conn, sql, params)
        done += chunk
        if done % (batch * 20) == 0 or done == total:
            log("  loaded %d/%d rows into %s" % (done, total, table))


def make_data(args):
    rng = np.random.default_rng(args.seed)
    vectors = rng.standard_normal((args.rows, args.dim), dtype=np.float32)
    queries = rng.standard_normal((args.queries, args.dim), dtype=np.float32)
    return vectors, queries


def ground_truth(vectors, queries, topk, metric):
    truth = []
    if metric == "l2":
        norms = (vectors * vectors).sum(axis=1)
        for q in queries:
            dist = norms + float((q * q).sum()) - 2.0 * vectors.dot(q)
            idx = np.argpartition(dist, topk - 1)[:topk]
            truth.append(set(int(i) for i in idx))
    else:
        norms = np.linalg.norm(vectors, axis=1)
        qnorms = np.linalg.norm(queries, axis=1)
        for qi, q in enumerate(queries):
            denom = norms * qnorms[qi]
            denom[denom == 0.0] = 1.0
            dist = 1.0 - vectors.dot(q) / denom
            idx = np.argpartition(dist, topk - 1)[:topk]
            truth.append(set(int(i) for i in idx))
    return truth


def run_queries(conn, table, queries, topk, metric, warmup, rounds):
    func = METRIC_SQL[metric][1]
    sql = ("SELECT id FROM `%s` ORDER BY %s(vec, %%s) APPROXIMATE LIMIT %d"
           % (table, func, topk))
    for i in range(warmup):
        execute(conn, sql, (vector_literal(queries[i % len(queries)]),))
    latencies = []
    results = []
    for _ in range(rounds):
        for q in queries:
            start = time.perf_counter()
            rows = execute(conn, sql, (vector_literal(q),))
            latencies.append((time.perf_counter() - start) * 1000.0)
            results.append(set(int(r[0]) for r in rows))
    latencies.sort()
    return {
        "count": len(latencies),
        "mean_ms": statistics.fmean(latencies),
        "p50_ms": latencies[len(latencies) // 2],
        "p95_ms": latencies[int(len(latencies) * 0.95) - 1],
        "max_ms": latencies[-1],
    }, results


def recall_at_k(results, truth, topk):
    if not results:
        return 0.0
    hits = 0
    for got, want in zip(results, truth):
        hits += len(got & want)
    return hits / float(len(results) * topk)


class GpuSampler(threading.Thread):
    """Samples rocm-smi while the index build is running."""

    def __init__(self):
        super().__init__(daemon=True)
        self.samples = []
        self.error = None
        self._stop_event = threading.Event()

    def run(self):
        while not self._stop_event.is_set():
            try:
                out = os.popen("rocm-smi --showuse --showmemuse 2>/dev/null").read()
                self.samples.append(out)
            except Exception as exc:  # pragma: no cover
                self.error = str(exc)
                return
            self._stop_event.wait(1.0)

    def stop(self):
        self._stop_event.set()
        self.join(timeout=5)

    def summary(self):
        text = "".join(self.samples)
        if not text.strip():
            return {"available": False, "note": "rocm-smi produced no output"}
        gpu_use = []
        for line in text.splitlines():
            if "GPU use" in line or "GPU utilization" in line.lower():
                # rocm-smi text output looks like "GPU[0] : GPU use (%): 17".
                match = re.search(r"GPU use \(%\):\s*(\d+(?:\.\d+)?)", line)
                if match:
                    gpu_use.append(float(match.group(1)))
                    continue
                for token in line.replace("|", " ").split():
                    if token.endswith("%"):
                        try:
                            gpu_use.append(float(token[:-1]))
                        except ValueError:
                            pass
        return {
            "available": True,
            "max_gpu_use_pct": max(gpu_use) if gpu_use else None,
            "samples": len(self.samples),
            "raw_tail": text[-1200:],
        }


def create_index(conn, table, lib, args):
    distance = METRIC_SQL[args.metric][0]
    sql = (
        "CREATE VECTOR INDEX idx_vec ON `%s` (vec) WITH "
        "(DISTANCE=%s, TYPE=HNSW, LIB=%s, M=%d, EF_CONSTRUCTION=%d, EF_SEARCH=%d)"
        % (table, distance, lib.upper(), args.m, args.ef_construction, args.ef_search)
    )
    start = time.perf_counter()
    execute(conn, sql)
    return time.perf_counter() - start


def prepare_base(conn, args, vectors):
    """Loads the shared data set once; every backend copies from it."""
    drop_table(conn, "bench_base")
    execute(
        conn,
        "CREATE TABLE `bench_base` (id INT PRIMARY KEY, vec VECTOR(%d), tag INT) "
        "ORGANIZATION = HEAP" % args.dim,
    )
    ids = list(range(args.rows))
    tags = [i % 64 for i in ids]
    start = time.perf_counter()
    bulk_insert(conn, "bench_base", ids, vectors, tags, args.batch)
    load_sec = time.perf_counter() - start
    execute(conn, "SELECT COUNT(*) FROM `bench_base`")
    return load_sec


def run_backend(conn, args, queries, truth, lib):
    result = {"lib": lib, "error": None, "build_sec": [], "load_sec": None}
    table = "bench_%s" % lib
    try:
        drop_table(conn, table)
        execute(
            conn,
            "CREATE TABLE `%s` (id INT PRIMARY KEY, vec VECTOR(%d), tag INT) "
            "ORGANIZATION = HEAP" % (table, args.dim),
        )
        start = time.perf_counter()
        execute(conn, "INSERT INTO `%s` SELECT id, vec, tag FROM bench_base" % table)
        result["load_sec"] = time.perf_counter() - start
        execute(conn, "SELECT COUNT(*) FROM `%s`" % table)

        if lib == "none":
            return result

        sampler = GpuSampler()
        sampler.start()
        try:
            for run in range(args.runs):
                start = time.perf_counter()
                build_sec = create_index(conn, table, lib, args)
                result["build_sec"].append(build_sec)
                log("  [%s] build run %d: %.2fs (statement %.2fs)"
                    % (lib, run + 1, time.perf_counter() - start, build_sec))
                if run + 1 < args.runs:
                    execute(conn, "DROP INDEX idx_vec ON `%s`" % table)
        finally:
            sampler.stop()
        result["gpu"] = sampler.summary()
        result["build_sec_median"] = statistics.median(result["build_sec"])
        result["build_sec_min"] = min(result["build_sec"])

        log("  [%s] measuring latency and recall" % lib)
        latency, got = run_queries(conn, table, queries, args.topk, args.metric,
                                   args.warmup, args.rounds)
        result["latency"] = latency
        result["recall_at_k"] = recall_at_k(got, truth, args.topk)
    except Exception as exc:
        result["error"] = "%s: %s" % (type(exc).__name__, exc)
        log("  [%s] failed: %s" % (lib, result["error"]))
        log(traceback.format_exc())
    finally:
        try:
            drop_table(conn, table)
        except Exception:
            pass
    return result


def collect_env(conn, args):
    env = {
        "server_version": None,
        "host": args.host,
        "port": args.port,
        "rows": args.rows,
        "dim": args.dim,
        "queries": args.queries,
        "topk": args.topk,
        "metric": args.metric,
        "m": args.m,
        "ef_construction": args.ef_construction,
        "ef_search": args.ef_search,
        "runs": args.runs,
        "rounds": args.rounds,
        "uname": os.uname().sysname + " " + os.uname().release,
    }
    try:
        rows = execute(conn, "SELECT @@version_comment")
        if rows:
            env["server_version"] = str(rows[0][0])
    except Exception:
        pass
    try:
        env["rocm_smi"] = os.popen("rocm-smi --showproductname 2>/dev/null").read()[-1200:]
    except Exception:
        env["rocm_smi"] = ""
    return env


def write_report(path, env, results):
    lines = []
    lines.append("# hipVS HNSW build benchmark\n")
    lines.append("Generated: %s\n" % time.strftime("%Y-%m-%d %H:%M:%S"))
    lines.append("## Environment\n")
    for key in ("server_version", "uname", "host", "port"):
        lines.append("- %s: `%s`" % (key, env.get(key)))
    lines.append("")
    lines.append("## Configuration\n")
    for key in ("rows", "dim", "queries", "topk", "metric", "m",
                "ef_construction", "ef_search", "runs", "rounds"):
        lines.append("- %s: `%s`" % (key, env.get(key)))
    lines.append("")
    lines.append("## Build (CREATE VECTOR INDEX over %d rows)\n" % env["rows"])
    if env.get("base_load_sec") is not None:
        lines.append("Base table load (no index): **%.2fs**\n" % env["base_load_sec"])
    lines.append("| LIB | build median (s) | build min (s) | build runs (s) | "
                 "load (s) | GPU peak (%%) | recall@%d | p50 (ms) | p95 (ms) |"
                 % env["topk"])
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for res in results:
        if res.get("error"):
            lines.append("| %s | error: %s | | | | | | | |"
                         % (res["lib"], res["error"].replace("|", " ")))
            continue
        build_runs = ", ".join("%.2f" % v for v in res.get("build_sec", []))
        gpu = res.get("gpu") or {}
        gpu_peak = gpu.get("max_gpu_use_pct")
        latency = res.get("latency") or {}
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s | %s | %s |"
            % (
                res["lib"],
                "-" if "build_sec_median" not in res else "%.2f" % res["build_sec_median"],
                "-" if "build_sec_min" not in res else "%.2f" % res["build_sec_min"],
                build_runs or "-",
                "-" if res["load_sec"] is None else "%.2f" % res["load_sec"],
                "-" if gpu_peak is None else "%.1f" % gpu_peak,
                "-" if "recall_at_k" not in res else "%.4f" % res["recall_at_k"],
                "-" if not latency else "%.2f" % latency["p50_ms"],
                "-" if not latency else "%.2f" % latency["p95_ms"],
            )
        )
    lines.append("")
    lines.append("Add the reproduction command, hardware facts and the GPU "
                 "evidence snippet (rocm-smi tail) below before submitting.\n")
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
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--runs", type=int, default=3, help="index build repetitions")
    parser.add_argument("--libs", default="none,vsag,hipvs")
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--query-timeout", type=int, default=3600)
    parser.add_argument("--output", default="hipvs_bench.json")
    parser.add_argument("--report", default="hipvs_report.md")
    return parser.parse_args()


def main():
    args = parse_args()
    libs = [v.strip().lower() for v in args.libs.split(",") if v.strip()]
    log("benchmark config: rows=%d dim=%d libs=%s" % (args.rows, args.dim, libs))

    vectors, queries = make_data(args)
    log("computing ground truth")
    truth = ground_truth(vectors, queries, args.topk, args.metric)

    conn = connect(args)
    try:
        execute(conn, "CREATE DATABASE IF NOT EXISTS `%s`" % args.database)
        execute(conn, "USE `%s`" % args.database)
        env = collect_env(conn, args)
        log("loading the shared base table")
        env["base_load_sec"] = prepare_base(conn, args, vectors)
        results = []
        for lib in libs:
            log("running backend %s" % lib)
            results.append(run_backend(conn, args, queries, truth, lib))
        drop_table(conn, "bench_base")
    finally:
        conn.close()

    payload = {"env": env, "results": results}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    log("wrote %s" % args.output)
    write_report(args.report, env, results)

    failed = [r for r in results if r.get("error")]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
