/*
 * Copyright (c) 2025 OceanBase.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#define USING_LOG_PREFIX LIB

#include "ob_hipvs_bridge.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <mutex>
#include <new>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

#include <vsag/allocator.h>

#include "lib/ob_define.h"
#include "lib/oblog/ob_log.h"
#include "lib/utility/ob_print_utils.h"

#ifdef OB_USE_HIPVS
#include <pthread.h>

#include <cuvs/core/c_api.h>
#include <cuvs/distance/distance.h>
#include <cuvs/neighbors/cagra.h>
#include <dlpack/dlpack.h>
#endif

namespace oceanbase {
namespace common {
namespace obhipvs {

namespace {

constexpr char HIPVS_STREAM_MAGIC[8] = {'O', 'B', 'H', 'I', 'P', 'V', 'S', '1'};
constexpr uint32_t HIPVS_STREAM_VERSION = 1;
// Extra candidates fetched per top-k so that the client side filter still has
// enough rows to return.
constexpr int64_t FILTER_OVERFETCH = 4;
constexpr int64_t MAX_SEARCH_CANDIDATES = 65536;
// Below this many rows the host side linear scan is cheaper than a GPU build
// and IVF-PQ, the CAGRA graph builder, is not meant for toy datasets.
constexpr int64_t MIN_GRAPH_ROWS = 1024;

enum MetricCode : int32_t {
  METRIC_L2 = 0,
  METRIC_IP = 1,
  METRIC_COSINE = 2,
  METRIC_INVALID = -1,
};

int to_metric_code(const char *metric)
{
  int code = METRIC_INVALID;
  if (metric != nullptr) {
    if (0 == strcmp(metric, "l2")) {
      code = METRIC_L2;
    } else if (0 == strcmp(metric, "ip")) {
      code = METRIC_IP;
    } else if (0 == strcmp(metric, "cosine")) {
      code = METRIC_COSINE;
    }
  }
  return code;
}

const char *metric_name(int32_t code)
{
  const char *name = "unknown";
  switch (code) {
    case METRIC_L2: name = "l2"; break;
    case METRIC_IP: name = "ip"; break;
    case METRIC_COSINE: name = "cosine"; break;
    default: break;
  }
  return name;
}

float compute_distance(int32_t metric, const float *lhs, const float *rhs, int32_t dim)
{
  float distance = 0.0f;
  if (metric == METRIC_L2) {
    for (int32_t i = 0; i < dim; ++i) {
      const float diff = lhs[i] - rhs[i];
      distance += diff * diff;
    }
  } else if (metric == METRIC_IP) {
    float dot = 0.0f;
    for (int32_t i = 0; i < dim; ++i) {
      dot += lhs[i] * rhs[i];
    }
    distance = -dot;
  } else if (metric == METRIC_COSINE) {
    float dot = 0.0f;
    float lhs_norm = 0.0f;
    float rhs_norm = 0.0f;
    for (int32_t i = 0; i < dim; ++i) {
      dot += lhs[i] * rhs[i];
      lhs_norm += lhs[i] * lhs[i];
      rhs_norm += rhs[i] * rhs[i];
    }
    const float denom = std::sqrt(lhs_norm) * std::sqrt(rhs_norm);
    distance = denom > 0.0f ? 1.0f - dot / denom : 1.0f;
  }
  return distance;
}

int64_t elapsed_us(const std::chrono::steady_clock::time_point &start)
{
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::steady_clock::now() - start)
      .count();
}

}  // namespace

class CagraIndex {
public:
  CagraIndex() = default;
  ~CagraIndex() = default;

  int64_t count() const { return static_cast<int64_t>(vids_.size()); }
  int32_t metric_code() const { return to_metric_code(config_.metric); }

  Config config_;
  std::vector<float> data_;
  std::vector<int64_t> vids_;
  std::vector<char> extras_;
  std::unordered_map<int64_t, int64_t> vid2row_;
  bool built_ = false;
  int64_t min_vid_ = INT64_MAX;
  int64_t max_vid_ = INT64_MIN;
  // Readers (search, distance lookups, serialization) share the lock, writers
  // (append, graph build, load) take it exclusively. The adaptor already holds
  // its own read/write lock, but the DDL append path does not, and CAGRA state
  // must never be rebuilt underneath a running search.
  std::shared_mutex rw_mutex_;
#ifdef OB_USE_HIPVS
  cuvsResources_t resources_ = 0;
  cuvsCagraIndex_t cagra_ = nullptr;
  struct Scratch {
    void *queries_ = nullptr;
    void *neighbors_ = nullptr;
    void *distances_ = nullptr;
    int64_t query_bytes_ = 0;
    int64_t result_bytes_ = 0;
    // cuVS resources own a stream and a workspace, so every concurrently used
    // scratch slot carries its own handle.
    cuvsResources_t resources_ = 0;
  };
  // One index can be searched concurrently, the scratch buffers are pooled and
  // lent out for the duration of a single search.
  std::mutex scratch_mutex_;
  std::vector<Scratch *> free_scratch_;
  std::vector<Scratch *> all_scratch_;
#endif
};

namespace {

#ifdef OB_USE_HIPVS

int cuvs_errcode2ob(cuvsError_t err, const char *action)
{
  int ret = OB_SUCCESS;
  if (err != CUVS_SUCCESS) {
    ret = OB_ERR_VSAG_RETURN_ERROR;
    const char *detail = cuvsGetLastErrorText();
    LOG_WARN("hipvs call failed", K(ret), KCSTRING(action),
             KCSTRING(detail == nullptr ? "" : detail));
  }
  return ret;
}

cuvsDistanceType to_cuvs_metric(int32_t code)
{
  cuvsDistanceType metric = L2Expanded;
  switch (code) {
    case METRIC_L2: metric = L2Expanded; break;
    case METRIC_IP: metric = InnerProduct; break;
    case METRIC_COSINE: metric = CosineExpanded; break;
    default: metric = L2Expanded; break;
  }
  return metric;
}

// The cuVS C API takes DLManagedTensor, the wrapper is borrowed only for the
// duration of the call so no deleter is installed.
void init_managed_tensor(DLManagedTensor &tensor, void *data, DLDeviceType device,
                         DLDataType dtype, int64_t rows, int64_t cols, int64_t *shape)
{
  memset(&tensor, 0, sizeof(tensor));
  shape[0] = rows;
  shape[1] = cols;
  tensor.dl_tensor.data = data;
  tensor.dl_tensor.device = DLDevice{device, 0};
  tensor.dl_tensor.ndim = 2;
  tensor.dl_tensor.dtype = dtype;
  tensor.dl_tensor.shape = shape;
  tensor.dl_tensor.strides = nullptr;
  tensor.dl_tensor.byte_offset = 0;
  tensor.manager_ctx = nullptr;
  tensor.deleter = nullptr;
}

void fill_float_tensor(DLManagedTensor &tensor, void *data, DLDeviceType device, int64_t rows,
                       int64_t cols, int64_t *shape)
{
  init_managed_tensor(tensor, data, device, DLDataType{kDLFloat, 32, 1}, rows, cols, shape);
}

void fill_u32_tensor(DLManagedTensor &tensor, void *data, DLDeviceType device, int64_t rows,
                     int64_t cols, int64_t *shape)
{
  init_managed_tensor(tensor, data, device, DLDataType{kDLUInt, 32, 1}, rows, cols, shape);
}

int ensure_scratch(CagraIndex *index, int64_t query_bytes, int64_t result_bytes,
                   CagraIndex::Scratch *&scratch)
{
  int ret = OB_SUCCESS;
  scratch = nullptr;
  if (OB_ISNULL(index)) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    std::lock_guard<std::mutex> guard(index->scratch_mutex_);
    for (int64_t i = 0; i < static_cast<int64_t>(index->free_scratch_.size()); ++i) {
      CagraIndex::Scratch *candidate = index->free_scratch_.at(i);
      if (candidate->query_bytes_ >= query_bytes && candidate->result_bytes_ >= result_bytes) {
        index->free_scratch_.erase(index->free_scratch_.begin() + i);
        scratch = candidate;
        break;
      }
    }
    if (scratch == nullptr) {
      CagraIndex::Scratch *fresh = new (std::nothrow) CagraIndex::Scratch();
      if (OB_ISNULL(fresh)) {
        ret = OB_ALLOCATE_MEMORY_FAILED;
      } else if (cudaSuccess != cudaMalloc(&fresh->queries_, query_bytes)
                 || cudaSuccess != cudaMalloc(&fresh->neighbors_, result_bytes)
                 || cudaSuccess != cudaMalloc(&fresh->distances_, result_bytes)
                 || OB_FAIL(cuvs_errcode2ob(cuvsResourcesCreate(&fresh->resources_),
                                            "create cuvs search resources"))) {
        if (fresh->queries_ != nullptr) {
          cudaFree(fresh->queries_);
        }
        if (fresh->neighbors_ != nullptr) {
          cudaFree(fresh->neighbors_);
        }
        if (fresh->distances_ != nullptr) {
          cudaFree(fresh->distances_);
        }
        if (fresh->resources_ != 0) {
          cuvsResourcesDestroy(fresh->resources_);
        }
        delete fresh;
        ret = OB_ALLOCATE_MEMORY_FAILED;
        LOG_WARN("hipvs device allocation failed", K(ret), K(query_bytes), K(result_bytes));
      } else {
        fresh->query_bytes_ = query_bytes;
        fresh->result_bytes_ = result_bytes;
        index->all_scratch_.push_back(fresh);
        scratch = fresh;
      }
    }
  }
  return ret;
}

void release_scratch(CagraIndex *index, CagraIndex::Scratch *scratch)
{
  if (OB_NOT_NULL(index) && OB_NOT_NULL(scratch)) {
    std::lock_guard<std::mutex> guard(index->scratch_mutex_);
    index->free_scratch_.push_back(scratch);
  }
}

// OB worker threads, the DDL pool in particular, run with the configured
// stack_size which defaults to 256KB. The cuVS CAGRA host side needs far more
// than that, so GPU work is dispatched to a private thread with a generous
// stack. tl_on_gpu_thread keeps nested calls (extend -> rebuild) inline.
constexpr size_t HIPVS_GPU_STACK_SIZE = 64L * 1024 * 1024;
thread_local bool tl_on_gpu_thread = false;

int run_on_gpu_thread(int (*fn)(void *), void *arg)
{
  int ret = OB_SUCCESS;
  if (tl_on_gpu_thread) {
    ret = fn(arg);
  } else {
    struct Task {
      int (*fn_)(void *);
      void *arg_;
      int ret_;
      int device_err_;
    } task = {fn, arg, OB_ERR_UNEXPECTED, 0};
    pthread_attr_t attr;
    pthread_t tid;
    int pret = pthread_attr_init(&attr);
    if (pret != 0) {
      ret = OB_ERR_UNEXPECTED;
    } else {
      pret = pthread_attr_setstacksize(&attr, HIPVS_GPU_STACK_SIZE);
      if (pret != 0) {
        ret = OB_ERR_UNEXPECTED;
      } else if (0 != (pret = pthread_create(
                           &tid, &attr,
                           [](void *p) -> void * {
                             Task *t = static_cast<Task *>(p);
                             tl_on_gpu_thread = true;
                             const cudaError_t err = cudaSetDevice(0);
                             if (err != cudaSuccess) {
                               t->ret_ = OB_ERR_UNEXPECTED;
                               t->device_err_ = static_cast<int>(err);
                             } else {
                               t->ret_ = t->fn_(t->arg_);
                             }
                             return nullptr;
                           },
                           &task))) {
        ret = OB_ERR_UNEXPECTED;
      } else {
        pthread_join(tid, nullptr);
        ret = task.ret_;
        if (task.device_err_ != 0) {
          LOG_WARN("hipvs set device on gpu thread failed", K(task.device_err_),
                   KCSTRING(cudaGetErrorString(static_cast<cudaError_t>(task.device_err_))));
        }
      }
      pthread_attr_destroy(&attr);
    }
    if (pret != 0) {
      LOG_WARN("hipvs run on gpu thread failed", K(ret), K(pret));
    }
  }
  return ret;
}

int build_graph_impl(CagraIndex *index)
{
  int ret = OB_SUCCESS;
  if (OB_ISNULL(index)) {
    ret = OB_INVALID_ARGUMENT;
  } else if (index->count() == 0) {
    index->built_ = false;
  } else if (index->count() < MIN_GRAPH_ROWS) {
    // Keep serving the small dataset from the host side linear scan.
    index->built_ = false;
  } else {
    const int64_t total = index->count();
    const int64_t max_degree = std::max<int64_t>(1, total - 1);
    const size_t graph_degree = static_cast<size_t>(
        std::min<int64_t>(index->config_.graph_degree > 0 ? index->config_.graph_degree : 32,
                          max_degree));
    const size_t intermediate_degree = static_cast<size_t>(
        std::max<int64_t>(static_cast<int64_t>(graph_degree) * 2,
                          index->config_.intermediate_graph_degree));
    cuvsCagraIndexParams_t params = nullptr;
    cuvsError_t err = cuvsCagraIndexParamsCreate(&params);
    if (OB_FAIL(cuvs_errcode2ob(err, "create cagra index params"))) {
    } else if (params == nullptr) {
      ret = OB_ERR_UNEXPECTED;
    } else {
      params->metric = to_cuvs_metric(index->metric_code());
      params->graph_degree = graph_degree;
      params->intermediate_graph_degree = std::max(intermediate_degree, graph_degree);
      params->build_algo = IVF_PQ;
      int64_t shape[2] = {0, 0};
      DLManagedTensor dataset_tensor;
      fill_float_tensor(dataset_tensor, const_cast<float *>(index->data_.data()), kDLCPU, total,
                        index->config_.dim, shape);
      // cuvsCagraBuild allocates a new index and stores it in the handle, the
      // previous one (if any) would leak unless it is destroyed here.
      if (index->cagra_ != nullptr) {
        cuvsCagraIndexDestroy(index->cagra_);
        index->cagra_ = nullptr;
      }
      // Between the destroy above and a successful build there is no usable
      // graph, search must not look at the handle until built_ flips back.
      index->built_ = false;
      if (OB_FAIL(cuvs_errcode2ob(cuvsCagraIndexCreate(&index->cagra_), "create cagra index"))) {
      } else if (OB_FAIL(cuvs_errcode2ob(
                     cuvsCagraBuild(index->resources_, params, &dataset_tensor, index->cagra_),
                     "cagra build"))) {
        cuvsCagraIndexDestroy(index->cagra_);
        index->cagra_ = nullptr;
      } else {
        index->built_ = true;
      }
      cuvsCagraIndexParamsDestroy(params);
    }
  }
  return ret;
}

int build_graph_task(void *arg) { return build_graph_impl(static_cast<CagraIndex *>(arg)); }

int build_graph(CagraIndex *index)
{
  int ret = OB_SUCCESS;
  if (OB_ISNULL(index)) {
    ret = OB_INVALID_ARGUMENT;
  } else if (index->count() < MIN_GRAPH_ROWS) {
    // Empty or tiny index, keep the host side linear scan, nothing to build.
    index->built_ = false;
  } else {
    ret = run_on_gpu_thread(build_graph_task, index);
  }
  return ret;
}

struct ExtendTask {
  CagraIndex *index_;
  const float *rows_;
  int64_t count_;
  int64_t old_count_;
  std::chrono::steady_clock::time_point start_;
};

// Always succeeds: the rows are already stored on the host, any GPU side
// failure only degrades search to the linear scan.
int extend_graph_impl(void *arg)
{
  int ret = OB_SUCCESS;
  ExtendTask *task = static_cast<ExtendTask *>(arg);
  CagraIndex *index = task->index_;
  cuvsCagraExtendParams_t params = nullptr;
  cuvsError_t err = cuvsCagraExtendParamsCreate(&params);
  if (OB_FAIL(cuvs_errcode2ob(err, "create cagra extend params"))) {
    LOG_WARN("hipvs extend params failed, keeping the host side scan", K(task->count_),
             K(task->old_count_));
    cuvsCagraIndexDestroy(index->cagra_);
    index->cagra_ = nullptr;
    index->built_ = false;
  } else {
    int64_t shape[2] = {0, 0};
    DLManagedTensor rows_tensor;
    fill_float_tensor(rows_tensor, const_cast<float *>(task->rows_), kDLCPU, task->count_,
                      index->config_.dim, shape);
    err = cuvsCagraExtend(index->resources_, params, &rows_tensor, index->cagra_);
    if (OB_FAIL(cuvs_errcode2ob(err, "cagra extend"))) {
      // The rows are already in the host buffers, rebuild so that the graph
      // and the dataset never drift apart.
      LOG_WARN("hipvs extend failed, rebuilding the graph", K(task->count_), K(task->old_count_));
      const int build_ret = build_graph(index);
      if (OB_SUCCESS != build_ret) {
        // build_graph has already dropped the handle and cleared built_, the
        // rows stay searchable through the host side scan instead of failing
        // the append and risking a duplicate retry.
        LOG_WARN("hipvs rebuild after a failed extend failed, keeping the host side scan",
                 K(build_ret), K(index->count()));
      }
    } else {
      LOG_INFO("hipvs extend finished", K(task->count_), K(index->count()), "cost_us",
               elapsed_us(task->start_));
    }
    cuvsCagraExtendParamsDestroy(params);
  }
  return OB_SUCCESS;
}

int extend_graph(CagraIndex *index, int64_t old_count, int64_t count)
{
  ExtendTask task;
  task.index_ = index;
  task.rows_ = index->data_.data() + old_count * index->config_.dim;
  task.count_ = count;
  task.old_count_ = old_count;
  task.start_ = std::chrono::steady_clock::now();
  return run_on_gpu_thread(extend_graph_impl, &task);
}

#endif  // OB_USE_HIPVS

int append_rows(CagraIndex *index, const float *vectors, const int64_t *ids, int64_t count,
                const char *extra_infos, int64_t &old_count)
{
  int ret = OB_SUCCESS;
  old_count = 0;
  if (OB_ISNULL(index) || count < 0 || (count > 0 && (OB_ISNULL(vectors) || OB_ISNULL(ids)))) {
    ret = OB_INVALID_ARGUMENT;
  } else if (count > 0) {
    const int32_t dim = index->config_.dim;
    const int32_t extra_size = index->config_.extra_info_size;
    old_count = index->count();
    try {
      // Only grow the map here, the vector inserts below are all or nothing
      // because their element types are trivially copyable.
      const size_t new_rows = static_cast<size_t>(old_count + count);
      if (new_rows
          > static_cast<size_t>(index->vid2row_.bucket_count()
                                * index->vid2row_.max_load_factor())) {
        index->vid2row_.reserve(new_rows * 2);
      }
      index->data_.insert(index->data_.end(), vectors, vectors + count * dim);
      index->vids_.insert(index->vids_.end(), ids, ids + count);
      if (extra_size > 0) {
        if (OB_ISNULL(extra_infos)) {
          index->extras_.resize(index->extras_.size() + count * extra_size, 0);
        } else {
          index->extras_.insert(index->extras_.end(), extra_infos,
                                extra_infos + count * extra_size);
        }
      }
      for (int64_t i = 0; i < count; ++i) {
        index->vid2row_[ids[i]] = old_count + i;
      }
    } catch (const std::exception &e) {
      ret = OB_ALLOCATE_MEMORY_FAILED;
      LOG_WARN("hipvs append rows failed", K(ret), KCSTRING(e.what()));
      // Roll the partial append back, the caller keeps the previous state.
      for (int64_t i = old_count; i < index->count(); ++i) {
        index->vid2row_.erase(index->vids_.at(i));
      }
      index->data_.resize(old_count * dim);
      if (extra_size > 0) {
        index->extras_.resize(old_count * extra_size);
      }
      index->vids_.resize(old_count);
      index->min_vid_ = INT64_MAX;
      index->max_vid_ = INT64_MIN;
      for (int64_t i = 0; i < old_count; ++i) {
        const int64_t vid = index->vids_.at(i);
        index->vid2row_[vid] = i;
        index->min_vid_ = std::min(index->min_vid_, vid);
        index->max_vid_ = std::max(index->max_vid_, vid);
      }
    }
    if (OB_SUCC(ret)) {
      for (int64_t i = 0; i < count; ++i) {
        const int64_t vid = ids[i];
        index->min_vid_ = std::min(index->min_vid_, vid);
        index->max_vid_ = std::max(index->max_vid_, vid);
      }
    }
  }
  return ret;
}

int write_stream(std::ostream &out, const void *data, int64_t bytes)
{
  int ret = OB_SUCCESS;
  if (bytes > 0) {
    out.write(static_cast<const char *>(data), bytes);
    if (!out.good()) {
      ret = OB_IO_ERROR;
    }
  }
  return ret;
}

int read_stream(std::istream &in, void *data, int64_t bytes)
{
  int ret = OB_SUCCESS;
  if (bytes > 0) {
    in.read(static_cast<char *>(data), bytes);
    if (!in.good()) {
      ret = OB_IO_ERROR;
    }
  }
  return ret;
}

}  // namespace

const Capability &capability()
{
  static std::once_flag once;
  static Capability cap;
  std::call_once(once, []() {
#ifdef OB_USE_HIPVS
    cap.compiled = true;
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count <= 0) {
      snprintf(cap.message, sizeof(cap.message), "no usable HIP device: %s",
               cudaGetErrorString(err));
    } else {
      cudaDeviceProp prop;
      memset(&prop, 0, sizeof(prop));
      if (cudaSuccess != cudaGetDeviceProperties(&prop, 0)) {
        snprintf(cap.message, sizeof(cap.message), "cannot query HIP device 0");
      } else {
        int runtime_version = 0;
        cudaRuntimeGetVersion(&runtime_version);
        cap.device_count = device_count;
        cap.runtime_version = runtime_version;
        snprintf(cap.device_name, sizeof(cap.device_name), "%s", prop.name);
        snprintf(cap.message, sizeof(cap.message), "available");
        cap.available = true;
      }
    }
#else
    cap.compiled = false;
    cap.available = false;
    snprintf(cap.message, sizeof(cap.message),
             "seekdb was built without hipVS, rebuild with -DOB_USE_HIPVS=ON");
#endif
  });
  return cap;
}

bool supports(int32_t index_type, const char *dtype, const char *metric)
{
  // obvsag::HNSW_TYPE == 0, obvsag::HGRAPH_TYPE == 6 (HNSW carrying extra
  // info). Only plain float32 HNSW shapes are mapped to CAGRA; the quantized
  // variants and IVF keep using the CPU implementation.
  return (index_type == 0 || index_type == 6) && dtype != nullptr
         && 0 == strcmp(dtype, "float32") && METRIC_INVALID != to_metric_code(metric);
}

int create(CagraIndex *&index, const Config &config)
{
  int ret = OB_SUCCESS;
  index = nullptr;
  const Capability &cap = capability();
  if (!cap.available) {
    ret = OB_NOT_SUPPORTED;
    LOG_WARN("hipvs backend is not available", K(ret), K(cap.compiled), KCSTRING(cap.message),
             KCSTRING(config.metric), K(config.dim), K(config.graph_degree));
  } else if (config.dim <= 0 || METRIC_INVALID == to_metric_code(config.metric)
             || config.extra_info_size < 0) {
    ret = OB_INVALID_ARGUMENT;
    LOG_WARN("invalid hipvs index config", K(ret), K(config.dim), KCSTRING(config.metric),
             K(config.extra_info_size));
  } else {
#ifdef OB_USE_HIPVS
    CagraIndex *fresh = new (std::nothrow) CagraIndex();
    if (OB_ISNULL(fresh)) {
      ret = OB_ALLOCATE_MEMORY_FAILED;
    } else {
      fresh->config_ = config;
      fresh->config_.metric = metric_name(to_metric_code(config.metric));
      const cudaError_t err = cudaSetDevice(0);
      if (err != cudaSuccess) {
        ret = OB_ERR_UNEXPECTED;
        LOG_WARN("hipvs set device failed", K(ret), KCSTRING(cudaGetErrorString(err)));
      } else if (OB_FAIL(cuvs_errcode2ob(cuvsResourcesCreate(&fresh->resources_),
                                         "create cuvs resources"))) {
      } else {
        LOG_INFO("hipvs index created", K(config.dim), KCSTRING(fresh->config_.metric),
                 K(config.graph_degree), K(config.ef_construction), K(config.extra_info_size),
                 KCSTRING(cap.device_name));
      }
      if (OB_FAIL(ret)) {
        if (fresh->resources_ != 0) {
          cuvsResourcesDestroy(fresh->resources_);
        }
        delete fresh;
        fresh = nullptr;
      } else {
        index = fresh;
      }
    }
#else
    ret = OB_NOT_SUPPORTED;
#endif
  }
  return ret;
}

void destroy(CagraIndex *&index)
{
  if (index != nullptr) {
    // Wait for running searches before tearing the device buffers down.
    std::unique_lock<std::shared_mutex> guard(index->rw_mutex_);
#ifdef OB_USE_HIPVS
    if (index->cagra_ != nullptr) {
      cuvsCagraIndexDestroy(index->cagra_);
      index->cagra_ = nullptr;
    }
    {
      std::lock_guard<std::mutex> guard(index->scratch_mutex_);
      for (int64_t i = 0; i < static_cast<int64_t>(index->all_scratch_.size()); ++i) {
        CagraIndex::Scratch *scratch = index->all_scratch_.at(i);
        if (scratch->queries_ != nullptr) {
          cudaFree(scratch->queries_);
        }
        if (scratch->neighbors_ != nullptr) {
          cudaFree(scratch->neighbors_);
        }
        if (scratch->distances_ != nullptr) {
          cudaFree(scratch->distances_);
        }
        if (scratch->resources_ != 0) {
          cuvsResourcesDestroy(scratch->resources_);
          scratch->resources_ = 0;
        }
        delete scratch;
      }
      index->all_scratch_.clear();
      index->free_scratch_.clear();
    }
    if (index->resources_ != 0) {
      cuvsResourcesDestroy(index->resources_);
      index->resources_ = 0;
    }
#endif
    delete index;
    index = nullptr;
  }
}

int build(CagraIndex *index, const float *vectors, const int64_t *ids, int64_t count,
          const char *extra_infos)
{
  int ret = OB_SUCCESS;
  int64_t old_count = 0;
  if (OB_ISNULL(index)) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    std::unique_lock<std::shared_mutex> guard(index->rw_mutex_);
    if (index->built_) {
      ret = OB_ERR_UNEXPECTED;
      LOG_WARN("hipvs index is built twice", K(ret), K(index->count()));
    } else if (OB_FAIL(append_rows(index, vectors, ids, count, extra_infos, old_count))) {
    } else {
#ifdef OB_USE_HIPVS
      const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
      if (OB_FAIL(build_graph(index))) {
      } else {
        LOG_INFO("hipvs build finished", K(ret), K(count), K(index->config_.dim),
                 KCSTRING(metric_name(index->metric_code())), K(index->config_.graph_degree),
                 "cost_us", elapsed_us(start), KCSTRING(capability().device_name));
      }
#else
      ret = OB_NOT_SUPPORTED;
#endif
    }
  }
  return ret;
}

// Turns a fully populated host side index into a searchable CAGRA graph. The
// DDL backfill only appends rows, the graph is built once the snapshot is
// complete (serialization) or when a snapshot without a graph is loaded.
int optimize(CagraIndex *index)
{
  int ret = OB_SUCCESS;
  if (OB_ISNULL(index)) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    std::unique_lock<std::shared_mutex> guard(index->rw_mutex_);
    if (index->built_ || index->count() == 0) {
      // nothing to do
    } else {
#ifdef OB_USE_HIPVS
      const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
      if (OB_FAIL(build_graph(index))) {
        // Keeping the rows searchable through the host side scan is better
        // than failing the load/seal that called optimize.
        LOG_WARN("hipvs graph build failed, keeping the host side scan", K(ret),
                 K(index->count()));
        ret = OB_SUCCESS;
      } else {
        LOG_INFO("hipvs graph built", K(ret), K(index->count()), K(index->config_.dim),
                 KCSTRING(metric_name(index->metric_code())), K(index->config_.graph_degree),
                 "cost_us", elapsed_us(start), KCSTRING(capability().device_name));
      }
#else
      ret = OB_NOT_SUPPORTED;
#endif
    }
  }
  return ret;
}

int extend(CagraIndex *index, const float *vectors, const int64_t *ids, int64_t count,
           const char *extra_infos)
{
  int ret = OB_SUCCESS;
  int64_t old_count = 0;
  if (OB_ISNULL(index)) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    std::unique_lock<std::shared_mutex> guard(index->rw_mutex_);
    if (OB_FAIL(append_rows(index, vectors, ids, count, extra_infos, old_count))) {
    } else if (!index->built_ || count == 0) {
      // Rows are staged on the host, search falls back to a linear scan and
      // optimize()/serialize() builds the graph once the snapshot is complete.
    } else {
#ifdef OB_USE_HIPVS
      // Never fails hard: the rows are stored either way, a failure only drops
      // the graph and keeps the host side scan serving the index.
      extend_graph(index, old_count, count);
#else
      ret = OB_NOT_SUPPORTED;
#endif
    }
  }
  return ret;
}

// Rows must be ordered best first, distances are compared against the user
// threshold which, like vsag, expects smaller values to be better.
bool candidate_kept(CagraIndex *index, const SearchOptions &options, int64_t total,
                    int64_t row, float distance, int64_t &vid, const char *&extra)
{
  bool keep = false;
  const int32_t extra_size = index->config_.extra_info_size;
  if (row >= 0 && row < total) {
    vid = index->vids_.at(row);
    extra = extra_size > 0 ? index->extras_.data() + row * extra_size : nullptr;
    keep = options.filter.accept == nullptr
               ? true
               : options.filter.accept(options.filter.ctx, vid, extra);
    if (keep && distance > options.distance_threshold) {
      keep = false;
    }
  }
  return keep;
}

int64_t count_kept(CagraIndex *index, const SearchOptions &options, const int64_t *rows,
                   const float *distances, int64_t count)
{
  const int64_t total = index->count();
  int64_t kept = 0;
  for (int64_t i = 0; i < count && kept < options.topk; ++i) {
    int64_t vid = 0;
    const char *extra = nullptr;
    if (candidate_kept(index, options, total, rows[i], distances[i], vid, extra)) {
      ++kept;
    }
  }
  return kept;
}

int collect_results(CagraIndex *index, const SearchOptions &options,
                    const int64_t *rows, const float *distances, int64_t count,
                    SearchResult &result)
{
  int ret = OB_SUCCESS;
  const int64_t total = index->count();
  const int32_t extra_size = index->config_.extra_info_size;
  void *result_allocator = options.search_allocator != nullptr ? options.search_allocator
                                                               : index->config_.allocator;
  if (OB_ISNULL(result_allocator)) {
    ret = OB_NOT_SUPPORTED;
    LOG_WARN("hipvs search needs an allocator for the result buffers", K(ret));
  } else {
    vsag::Allocator *allocator = static_cast<vsag::Allocator *>(result_allocator);
    const int64_t max_out = std::max<int64_t>(1, std::min<int64_t>(count, options.topk));
    float *out_distances =
        static_cast<float *>(allocator->Allocate(sizeof(float) * max_out));
    int64_t *out_ids =
        static_cast<int64_t *>(allocator->Allocate(sizeof(int64_t) * max_out));
    char *out_extras = nullptr;
    if (options.need_extra_info && extra_size > 0) {
      out_extras = static_cast<char *>(allocator->Allocate(extra_size * max_out));
    }
    if (OB_ISNULL(out_distances) || OB_ISNULL(out_ids)
        || (options.need_extra_info && extra_size > 0 && OB_ISNULL(out_extras))) {
      ret = OB_ALLOCATE_MEMORY_FAILED;
      LOG_WARN("hipvs allocate result buffers failed", K(ret), K(max_out));
      if (OB_NOT_NULL(out_distances)) {
        allocator->Deallocate(out_distances);
      }
      if (OB_NOT_NULL(out_ids)) {
        allocator->Deallocate(out_ids);
      }
      if (OB_NOT_NULL(out_extras)) {
        allocator->Deallocate(out_extras);
      }
    } else {
      int64_t kept = 0;
      for (int64_t i = 0; i < count && kept < options.topk; ++i) {
        int64_t vid = 0;
        const char *extra = nullptr;
        if (candidate_kept(index, options, total, rows[i], distances[i], vid, extra)) {
          out_ids[kept] = vid;
          out_distances[kept] = distances[i];
          if (OB_NOT_NULL(out_extras)) {
            memcpy(out_extras + kept * extra_size, extra, extra_size);
          }
          ++kept;
        }
      }
      result.distances = out_distances;
      result.ids = out_ids;
      result.extra_infos = out_extras;
      result.count = kept;
    }
  }
  return ret;
}

// The incremental mem table is searched before its rows are merged into a
// snapshot and the CAGRA graph is only built on the snapshot, so freshly
// inserted rows need a host side scan. The mem table is small by construction.
int linear_search(CagraIndex *index, const float *query, const SearchOptions &options,
                  int64_t candidates, SearchResult &result)
{
  int ret = OB_SUCCESS;
  const int64_t total = index->count();
  const int32_t dim = index->config_.dim;
  const int32_t metric = index->metric_code();
  try {
    std::vector<float> all_distances(total);
    std::vector<int64_t> order(total);
    for (int64_t i = 0; i < total; ++i) {
      all_distances[i] = compute_distance(metric, query, index->data_.data() + i * dim, dim);
      order[i] = i;
    }
    std::partial_sort(order.begin(), order.begin() + candidates, order.end(),
                      [&all_distances](int64_t lhs, int64_t rhs) {
                        return all_distances[lhs] < all_distances[rhs];
                      });
    const int64_t kept = std::min(candidates, total);
    std::vector<float> top_distances(kept);
    for (int64_t i = 0; i < kept; ++i) {
      top_distances[i] = all_distances[order[i]];
    }
    ret = collect_results(index, options, order.data(), top_distances.data(), kept, result);
  } catch (const std::exception &e) {
    ret = OB_ALLOCATE_MEMORY_FAILED;
    LOG_WARN("hipvs linear scan allocation failed", K(ret), K(total), KCSTRING(e.what()));
  }
  return ret;
}

#ifdef OB_USE_HIPVS
// Runs one CAGRA search for a single query and copies the result back to the
// host. Rows are dataset row ids ordered best first and distances use the vsag
// convention, smaller is better. The caller holds the index read lock.
int device_search_window(CagraIndex *index, const float *query, int64_t window,
                         int32_t ef_search, std::vector<int64_t> &rows,
                         std::vector<float> &distances)
{
  int ret = OB_SUCCESS;
  const int32_t dim = index->config_.dim;
  CagraIndex::Scratch *scratch = nullptr;
  rows.clear();
  distances.clear();
  if (OB_FAIL(ensure_scratch(index, static_cast<int64_t>(sizeof(float)) * dim,
                             static_cast<int64_t>(sizeof(uint32_t)) * window, scratch))) {
  } else {
    std::vector<uint32_t> raw_ids;
    try {
      raw_ids.resize(window);
      rows.resize(window);
      distances.resize(window);
    } catch (const std::exception &e) {
      ret = OB_ALLOCATE_MEMORY_FAILED;
      LOG_WARN("hipvs allocate staging buffers failed", K(ret), KCSTRING(e.what()));
    }
    if (OB_SUCC(ret)
        && cudaSuccess
               != cudaMemcpy(scratch->queries_, query, sizeof(float) * dim,
                             cudaMemcpyHostToDevice)) {
      ret = OB_ERR_UNEXPECTED;
      LOG_WARN("hipvs copy query to device failed", K(ret));
    }
    if (OB_SUCC(ret)) {
      int64_t queries_shape[2] = {0, 0};
      int64_t neighbors_shape[2] = {0, 0};
      int64_t distances_shape[2] = {0, 0};
      DLManagedTensor queries_tensor;
      DLManagedTensor neighbors_tensor;
      DLManagedTensor distances_tensor;
      fill_float_tensor(queries_tensor, scratch->queries_, kDLCUDA, 1, dim, queries_shape);
      fill_u32_tensor(neighbors_tensor, scratch->neighbors_, kDLCUDA, 1, window,
                      neighbors_shape);
      fill_float_tensor(distances_tensor, scratch->distances_, kDLCUDA, 1, window,
                        distances_shape);
      cuvsCagraSearchParams_t params = nullptr;
      cuvsError_t err = cuvsCagraSearchParamsCreate(&params);
      if (OB_FAIL(cuvs_errcode2ob(err, "create cagra search params"))) {
      } else {
        // cuVS requires itopk_size >= k, and the beam has to be at least as
        // wide as the window we are paging through. A beam wider than the
        // dataset only adds never filled slots, so clamp it to the row count.
        const int64_t itopk = std::min<int64_t>(
            index->count(), std::max<int64_t>(window, ef_search > 0 ? ef_search : 64));
        params->itopk_size = static_cast<size_t>(itopk);
        cuvsFilter filter;
        filter.addr = 0;
        filter.type = NO_FILTER;
        err = cuvsCagraSearch(scratch->resources_, params, index->cagra_, &queries_tensor,
                              &neighbors_tensor, &distances_tensor, filter);
        if (OB_FAIL(cuvs_errcode2ob(err, "cagra search"))) {
        } else if (cudaSuccess
                       != cudaMemcpy(raw_ids.data(), scratch->neighbors_,
                                     sizeof(uint32_t) * window, cudaMemcpyDeviceToHost)
                   || cudaSuccess
                          != cudaMemcpy(distances.data(), scratch->distances_,
                                        sizeof(float) * window, cudaMemcpyDeviceToHost)) {
          ret = OB_ERR_UNEXPECTED;
          LOG_WARN("hipvs copy search result to host failed", K(ret));
        }
        cuvsCagraSearchParamsDestroy(params);
      }
    }
    release_scratch(index, scratch);
    if (OB_SUCC(ret)) {
      // CAGRA ranks inner product by similarity while the rest of the engine
      // compares distances and expects smaller values to be better.
      const bool negate = index->metric_code() == METRIC_IP;
      for (int64_t i = 0; i < window; ++i) {
        rows[i] = static_cast<int64_t>(raw_ids[i]);
        if (negate) {
          distances[i] = -distances[i];
        }
      }
    }
  }
  return ret;
}
#endif

int search(CagraIndex *index, const float *query, const SearchOptions &options,
           SearchResult &result)
{
  int ret = OB_SUCCESS;
  result = SearchResult();
  if (OB_ISNULL(index) || OB_ISNULL(query) || options.topk <= 0) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    std::shared_lock<std::shared_mutex> guard(index->rw_mutex_);
    if (index->count() == 0) {
      // An empty index has no neighbours, which is not an error.
    } else {
      const int64_t total = index->count();
      const bool has_filter = options.filter.accept != nullptr;
      const int64_t already_returned = std::max<int64_t>(0, options.already_returned);
      int64_t candidates = std::min(total, options.topk + already_returned);
      if (has_filter) {
        float ratio = options.filter.valid_ratio;
        if (!(ratio > 0.0f) || ratio > 1.0f) {
          ratio = 1.0f;
        }
        const int64_t wanted = static_cast<int64_t>(std::ceil(
                                   static_cast<double>(options.topk) * FILTER_OVERFETCH / ratio))
                               + already_returned;
        candidates = std::min(total, std::max(candidates, wanted));
      }
      candidates = std::min(candidates, MAX_SEARCH_CANDIDATES);
#ifdef OB_USE_HIPVS
      if (!index->built_) {
        // Nothing has built the graph yet, the rows are scanned on the host.
        // With a filter the whole set is ranked so that the filtered top-k is
        // not limited by an arbitrary window.
        ret = linear_search(index, query, options, has_filter ? total : candidates, result);
      } else {
        const int32_t ef_search =
            options.ef_search > 0 ? options.ef_search : index->config_.ef_search;
        std::vector<int64_t> rows;
        std::vector<float> distances;
        int64_t window = candidates;
        while (OB_SUCC(ret)) {
          if (OB_FAIL(device_search_window(index, query, window, ef_search, rows, distances))) {
          } else if (!has_filter || window >= total
                     || count_kept(index, options, rows.data(), distances.data(), window)
                            >= options.topk) {
            break;
          } else {
            const int64_t next = std::min<int64_t>(total, window * 2);
            if (next <= window) {
              break;
            }
            window = next;
          }
        }
        if (OB_SUCC(ret)) {
          ret = collect_results(index, options, rows.data(), distances.data(),
                                static_cast<int64_t>(rows.size()), result);
        }
      }
#else
      ret = OB_NOT_SUPPORTED;
      LOG_WARN("hipvs search called on a bridge built without hipVS", K(ret));
#endif
    }
  }
  return ret;
}

int calc_distance_by_id(CagraIndex *index, const float *query, const int64_t *ids,
                        int64_t count, const float *&distances)
{
  int ret = OB_SUCCESS;
  distances = nullptr;
  if (OB_ISNULL(index) || OB_ISNULL(query) || OB_ISNULL(ids) || count < 0) {
    ret = OB_INVALID_ARGUMENT;
  } else if (OB_ISNULL(index->config_.allocator)) {
    ret = OB_NOT_SUPPORTED;
    LOG_WARN("hipvs distance by id needs the index allocator", K(ret));
  } else if (count == 0) {
    // nothing to look up
  } else {
    std::shared_lock<std::shared_mutex> guard(index->rw_mutex_);
    vsag::Allocator *allocator = static_cast<vsag::Allocator *>(index->config_.allocator);
    float *buffer = static_cast<float *>(allocator->Allocate(sizeof(float) * count));
    if (OB_ISNULL(buffer)) {
      ret = OB_ALLOCATE_MEMORY_FAILED;
    } else {
      const int32_t dim = index->config_.dim;
      const int32_t metric = index->metric_code();
      // vsag reports a missing vid through a -1 distance, the caller uses that
      // to tell "not stored in this index" from a real neighbour.
      for (int64_t i = 0; i < count; ++i) {
        std::unordered_map<int64_t, int64_t>::const_iterator iter = index->vid2row_.find(ids[i]);
        if (iter == index->vid2row_.end()) {
          buffer[i] = -1.0f;
        } else {
          buffer[i] = compute_distance(metric, query,
                                       index->data_.data() + iter->second * dim, dim);
        }
      }
      distances = buffer;
    }
  }
  return ret;
}

int get_extra_info_by_ids(CagraIndex *index, const int64_t *ids, int64_t count,
                          char *extra_infos)
{
  int ret = OB_SUCCESS;
  if (OB_ISNULL(index) || OB_ISNULL(ids) || count < 0
      || (count > 0 && OB_ISNULL(extra_infos))) {
    ret = OB_INVALID_ARGUMENT;
  } else if (index->config_.extra_info_size <= 0) {
    ret = OB_NOT_SUPPORTED;
  } else {
    std::shared_lock<std::shared_mutex> guard(index->rw_mutex_);
    const int64_t extra_size = index->config_.extra_info_size;
    for (int64_t i = 0; i < count && OB_SUCC(ret); ++i) {
      std::unordered_map<int64_t, int64_t>::const_iterator iter = index->vid2row_.find(ids[i]);
      if (iter == index->vid2row_.end()) {
        ret = OB_ENTRY_NOT_EXIST;
        LOG_WARN("hipvs extra info misses the vid", K(ret), K(ids[i]));
      } else {
        memcpy(extra_infos + i * extra_size,
               index->extras_.data() + iter->second * extra_size, extra_size);
      }
    }
  }
  return ret;
}

int get_count(CagraIndex *index, int64_t &count)
{
  int ret = OB_SUCCESS;
  count = 0;
  if (OB_ISNULL(index)) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    std::shared_lock<std::shared_mutex> guard(index->rw_mutex_);
    count = index->count();
  }
  return ret;
}

int get_vid_bound(CagraIndex *index, int64_t &min_vid, int64_t &max_vid)
{
  int ret = OB_SUCCESS;
  min_vid = 0;
  max_vid = 0;
  if (OB_ISNULL(index)) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    std::shared_lock<std::shared_mutex> guard(index->rw_mutex_);
    if (index->count() > 0) {
      min_vid = index->min_vid_;
      max_vid = index->max_vid_;
    }
  }
  return ret;
}

uint64_t estimate_memory(const Config &config, int64_t count, bool is_build)
{
  const int64_t dim = config.dim > 0 ? config.dim : 0;
  const int64_t degree = config.graph_degree > 0 ? config.graph_degree : 32;
  const int64_t extra = config.extra_info_size > 0 ? config.extra_info_size : 0;
  // host dataset + host vid map + device copy of the dataset and the graph
  uint64_t bytes = static_cast<uint64_t>(count)
                   * (static_cast<uint64_t>(dim) * sizeof(float) + sizeof(int64_t) + extra
                      + static_cast<uint64_t>(degree) * sizeof(uint32_t));
  bytes += static_cast<uint64_t>(count) * static_cast<uint64_t>(dim) * sizeof(float);
  if (is_build) {
    bytes = static_cast<uint64_t>(static_cast<double>(bytes) * 1.5);
  }
  return bytes;
}

int serialize(CagraIndex *index, std::ostream &out)
{
  int ret = OB_SUCCESS;
  if (OB_ISNULL(index)) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    std::unique_lock<std::shared_mutex> guard(index->rw_mutex_);
    // The snapshot is written once the DDL backfill is complete, which makes it
    // the natural point to build the graph so that loading does not have to.
    // A failed build is only logged, the row data is still complete and the
    // index keeps working through the linear fallback.
    if (!index->built_ && index->count() > 0) {
#ifdef OB_USE_HIPVS
      const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
      int build_ret = build_graph(index);
      if (OB_SUCCESS != build_ret) {
        LOG_WARN("hipvs graph build before serialize failed", K(build_ret), K(index->count()));
      } else {
        LOG_INFO("hipvs graph built before serialize", K(index->count()), K(index->config_.dim),
                 "cost_us", elapsed_us(start), KCSTRING(capability().device_name));
      }
#else
      LOG_WARN("hipvs graph cannot be built before serialize without hipVS");
#endif
    }
    const int64_t count = index->count();
    const int32_t dim = index->config_.dim;
    const int32_t extra_size = index->config_.extra_info_size;
    const int32_t metric = index->metric_code();
    uint8_t has_graph = index->built_ ? 1 : 0;
    uint32_t graph_degree = 0;
    std::vector<uint32_t> host_graph;
#ifdef OB_USE_HIPVS
    if (index->built_ && index->cagra_ != nullptr) {
      DLManagedTensor graph_tensor;
      memset(&graph_tensor, 0, sizeof(graph_tensor));
      if (OB_FAIL(cuvs_errcode2ob(cuvsCagraIndexGetGraph(index->cagra_, &graph_tensor),
                                  "cagra get graph"))) {
      } else if (graph_tensor.dl_tensor.ndim != 2) {
        ret = OB_ERR_UNEXPECTED;
      } else {
        const int64_t rows = graph_tensor.dl_tensor.shape[0];
        const int64_t cols = graph_tensor.dl_tensor.shape[1];
        try {
          host_graph.resize(rows * cols);
        } catch (const std::exception &e) {
          ret = OB_ALLOCATE_MEMORY_FAILED;
          LOG_WARN("hipvs allocate graph buffer failed", K(ret), KCSTRING(e.what()));
        }
        if (OB_SUCC(ret)) {
          if (cudaSuccess
              != cudaMemcpy(host_graph.data(), graph_tensor.dl_tensor.data,
                            sizeof(uint32_t) * rows * cols, cudaMemcpyDeviceToHost)) {
            ret = OB_ERR_UNEXPECTED;
            LOG_WARN("hipvs copy graph to host failed", K(ret));
          } else {
            graph_degree = static_cast<uint32_t>(cols);
          }
        }
      }
      if (graph_tensor.deleter != nullptr) {
        graph_tensor.deleter(&graph_tensor);
      }
    }
#endif
    if (OB_SUCC(ret)) {
      if (OB_FAIL(write_stream(out, HIPVS_STREAM_MAGIC, sizeof(HIPVS_STREAM_MAGIC)))
          || OB_FAIL(write_stream(out, &HIPVS_STREAM_VERSION, sizeof(HIPVS_STREAM_VERSION)))
          || OB_FAIL(write_stream(out, &dim, sizeof(dim)))
          || OB_FAIL(write_stream(out, &metric, sizeof(metric)))
          || OB_FAIL(write_stream(out, &extra_size, sizeof(extra_size)))
          || OB_FAIL(write_stream(out, &count, sizeof(count)))
          || OB_FAIL(write_stream(out, &has_graph, sizeof(has_graph)))
          || OB_FAIL(write_stream(out, &graph_degree, sizeof(graph_degree)))) {
      } else if (OB_FAIL(write_stream(out, index->data_.data(),
                                      count * dim * sizeof(float)))
                 || OB_FAIL(write_stream(out, index->vids_.data(),
                                         count * sizeof(int64_t)))
                 || (extra_size > 0
                     && OB_FAIL(write_stream(out, index->extras_.data(),
                                             count * extra_size)))
                 || (graph_degree > 0
                     && OB_FAIL(write_stream(out, host_graph.data(),
                                             count * graph_degree * sizeof(uint32_t))))) {
      }
    }
  }
  return ret;
}

int deserialize(CagraIndex *&index, std::istream &in)
{
  int ret = OB_SUCCESS;
  if (OB_ISNULL(index)) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    char magic[sizeof(HIPVS_STREAM_MAGIC)];
    uint32_t version = 0;
    int32_t dim = 0;
    int32_t metric = 0;
    int32_t extra_size = 0;
    int64_t count = 0;
    uint8_t has_graph = 0;
    uint32_t graph_degree = 0;
    if (OB_FAIL(read_stream(in, magic, sizeof(magic)))
        || OB_FAIL(read_stream(in, &version, sizeof(version)))
        || OB_FAIL(read_stream(in, &dim, sizeof(dim)))
        || OB_FAIL(read_stream(in, &metric, sizeof(metric)))
        || OB_FAIL(read_stream(in, &extra_size, sizeof(extra_size)))
        || OB_FAIL(read_stream(in, &count, sizeof(count)))
        || OB_FAIL(read_stream(in, &has_graph, sizeof(has_graph)))
        || OB_FAIL(read_stream(in, &graph_degree, sizeof(graph_degree)))) {
    } else if (0 != memcmp(magic, HIPVS_STREAM_MAGIC, sizeof(magic))
               || version != HIPVS_STREAM_VERSION) {
      ret = OB_INVALID_DATA;
      LOG_WARN("hipvs stream header mismatch", K(ret), KP(magic), K(version));
    } else if (count < 0 || dim <= 0 || dim != index->config_.dim
               || extra_size != index->config_.extra_info_size
               || metric != index->metric_code()) {
      ret = OB_INVALID_DATA;
      LOG_WARN("hipvs stream params mismatch", K(ret), K(count), K(dim), K(extra_size),
               K(metric), K(index->config_.dim), K(index->config_.extra_info_size));
    } else {
      std::unique_lock<std::shared_mutex> guard(index->rw_mutex_);
      // Loading replaces whatever the index currently holds, including a graph
      // built by earlier appends. cuvsCagraIndexFromArgs overwrites the handle
      // storage, so the old handle has to be released first.
      index->built_ = false;
#ifdef OB_USE_HIPVS
      if (index->cagra_ != nullptr) {
        cuvsCagraIndexDestroy(index->cagra_);
        index->cagra_ = nullptr;
      }
#endif
      index->data_.clear();
      index->vids_.clear();
      index->extras_.clear();
      index->vid2row_.clear();
      index->min_vid_ = INT64_MAX;
      index->max_vid_ = INT64_MIN;
      if (count == 0) {
        // empty index
      } else {
        try {
          index->data_.resize(count * dim);
          index->vids_.resize(count);
          if (extra_size > 0) {
            index->extras_.resize(count * extra_size);
          }
        } catch (const std::exception &e) {
          ret = OB_ALLOCATE_MEMORY_FAILED;
          LOG_WARN("hipvs allocate deserialize buffers failed", K(ret), KCSTRING(e.what()));
        }
        if (OB_SUCC(ret)
            && (OB_FAIL(read_stream(in, index->data_.data(), count * dim * sizeof(float)))
                || OB_FAIL(read_stream(in, index->vids_.data(), count * sizeof(int64_t)))
                || (extra_size > 0
                    && OB_FAIL(read_stream(in, index->extras_.data(),
                                           count * extra_size))))) {
        }
        if (OB_SUCC(ret)) {
          for (int64_t i = 0; i < count; ++i) {
            const int64_t vid = index->vids_.at(i);
            index->vid2row_[vid] = i;
            index->min_vid_ = std::min(index->min_vid_, vid);
            index->max_vid_ = std::max(index->max_vid_, vid);
          }
        }
#ifdef OB_USE_HIPVS
        if (OB_SUCC(ret) && has_graph == 1 && graph_degree > 0) {
          std::vector<uint32_t> host_graph;
          try {
            host_graph.resize(count * graph_degree);
          } catch (const std::exception &e) {
            ret = OB_ALLOCATE_MEMORY_FAILED;
            LOG_WARN("hipvs allocate graph buffers failed", K(ret), KCSTRING(e.what()));
          }
          if (OB_SUCC(ret) && index->cagra_ == nullptr
              && OB_FAIL(cuvs_errcode2ob(cuvsCagraIndexCreate(&index->cagra_),
                                         "create cagra index"))) {
          }
          if (OB_SUCC(ret)
              && OB_FAIL(read_stream(in, host_graph.data(),
                                     count * graph_degree * sizeof(uint32_t)))) {
          }
          if (OB_SUCC(ret)) {
            int64_t dataset_shape[2] = {0, 0};
            int64_t graph_shape[2] = {0, 0};
            DLManagedTensor dataset_tensor;
            DLManagedTensor graph_tensor;
            fill_float_tensor(dataset_tensor, index->data_.data(), kDLCPU, count, dim,
                              dataset_shape);
            fill_u32_tensor(graph_tensor, host_graph.data(), kDLCPU, count, graph_degree,
                            graph_shape);
            if (OB_FAIL(cuvs_errcode2ob(cuvsCagraIndexFromArgs(index->resources_,
                                                               to_cuvs_metric(
                                                                   index->metric_code()),
                                                               &graph_tensor, &dataset_tensor,
                                                               index->cagra_),
                                        "cagra index from args"))) {
            } else {
              index->built_ = true;
            }
          }
          if (OB_FAIL(ret) && index->cagra_ != nullptr) {
            cuvsCagraIndexDestroy(index->cagra_);
            index->cagra_ = nullptr;
          }
        } else if (OB_SUCC(ret) && has_graph == 1) {
          ret = OB_INVALID_DATA;
          LOG_WARN("hipvs stream has a graph without degree", K(ret));
        }
#else
        if (OB_SUCC(ret) && has_graph == 1) {
          ret = OB_NOT_SUPPORTED;
          LOG_WARN("hipvs stream cannot be loaded without hipVS support", K(ret));
        }
#endif
      }
    }
  }
  return ret;
}

}  // namespace obhipvs
}  // namespace common
}  // namespace oceanbase
