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

#include "ob_vector_util.h"
#include <atomic>
#include <cmath>
#include <mutex>
#include <new>
#include <unordered_map>
#include <unordered_set>
#include "lib/ob_define.h"
#include "lib/oblog/ob_log.h"
#include "lib/string/ob_string.h"

namespace oceanbase {
namespace common {
namespace obvectorutil {

namespace {

// A LIB=HIPVS index is stored as this wrapper instead of the obvsag handle.
// Plain VSAG/OB indexes keep the original handle untouched, so the default
// path behaves exactly as before.
struct HipvsHandle {
  obhipvs::Config config_;
  obhipvs::CagraIndex *index_;
};

// Paging state for the iterative filtered search. CAGRA has no server side
// iterator, so the ids already returned are remembered and skipped.
struct HipvsIterator {
  std::unordered_set<int64_t> returned_;
};

struct HipvsFilterCtx {
  obvsag::FilterInterface *filter_;
  bool use_extra_info_;
  bool reverse_;
  std::unordered_set<int64_t> *returned_;
};

std::mutex &hipvs_registry_mutex()
{
  static std::mutex mutex;
  return mutex;
}

std::unordered_map<void *, HipvsHandle *> &hipvs_handles()
{
  static std::unordered_map<void *, HipvsHandle *> handles;
  return handles;
}

std::unordered_map<void *, HipvsIterator *> &hipvs_iterators()
{
  static std::unordered_map<void *, HipvsIterator *> iters;
  return iters;
}

std::atomic<int64_t> g_hipvs_handle_count(0);
std::atomic<int64_t> g_hipvs_iter_count(0);

HipvsHandle *find_hipvs_handle(void *ptr)
{
  HipvsHandle *handle = nullptr;
  if (OB_NOT_NULL(ptr) && g_hipvs_handle_count.load(std::memory_order_acquire) > 0) {
    std::lock_guard<std::mutex> guard(hipvs_registry_mutex());
    std::unordered_map<void *, HipvsHandle *>::const_iterator iter = hipvs_handles().find(ptr);
    if (iter != hipvs_handles().end()) {
      handle = iter->second;
    }
  }
  return handle;
}

void register_hipvs_handle(void *ptr, HipvsHandle *handle)
{
  std::lock_guard<std::mutex> guard(hipvs_registry_mutex());
  hipvs_handles()[ptr] = handle;
  g_hipvs_handle_count.fetch_add(1, std::memory_order_release);
}

HipvsHandle *take_hipvs_handle(void *ptr)
{
  HipvsHandle *handle = nullptr;
  if (OB_NOT_NULL(ptr) && g_hipvs_handle_count.load(std::memory_order_acquire) > 0) {
    std::lock_guard<std::mutex> guard(hipvs_registry_mutex());
    std::unordered_map<void *, HipvsHandle *>::iterator iter = hipvs_handles().find(ptr);
    if (iter != hipvs_handles().end()) {
      handle = iter->second;
      hipvs_handles().erase(iter);
      g_hipvs_handle_count.fetch_sub(1, std::memory_order_release);
    }
  }
  return handle;
}

HipvsIterator *find_hipvs_iterator(void *ptr)
{
  HipvsIterator *iter = nullptr;
  if (OB_NOT_NULL(ptr) && g_hipvs_iter_count.load(std::memory_order_acquire) > 0) {
    std::lock_guard<std::mutex> guard(hipvs_registry_mutex());
    std::unordered_map<void *, HipvsIterator *>::const_iterator found =
        hipvs_iterators().find(ptr);
    if (found != hipvs_iterators().end()) {
      iter = found->second;
    }
  }
  return iter;
}

void register_hipvs_iterator(HipvsIterator *iter)
{
  std::lock_guard<std::mutex> guard(hipvs_registry_mutex());
  hipvs_iterators()[static_cast<void *>(iter)] = iter;
  g_hipvs_iter_count.fetch_add(1, std::memory_order_release);
}

HipvsIterator *take_hipvs_iterator(void *ptr)
{
  HipvsIterator *iter = nullptr;
  if (OB_NOT_NULL(ptr) && g_hipvs_iter_count.load(std::memory_order_acquire) > 0) {
    std::lock_guard<std::mutex> guard(hipvs_registry_mutex());
    std::unordered_map<void *, HipvsIterator *>::iterator found = hipvs_iterators().find(ptr);
    if (found != hipvs_iterators().end()) {
      iter = found->second;
      hipvs_iterators().erase(found);
      g_hipvs_iter_count.fetch_sub(1, std::memory_order_release);
    }
  }
  return iter;
}

bool hipvs_filter_accept(void *ctx, int64_t id, const char *extra_info)
{
  bool keep = true;
  HipvsFilterCtx *filter_ctx = static_cast<HipvsFilterCtx *>(ctx);
  if (OB_ISNULL(filter_ctx)) {
    keep = true;
  } else if (OB_NOT_NULL(filter_ctx->returned_) &&
             filter_ctx->returned_->find(id) != filter_ctx->returned_->end()) {
    keep = false;
  } else if (OB_NOT_NULL(filter_ctx->filter_)) {
    // Mirrors ObVasgFilter: test() reports a hit and reverse flips the verdict.
    const bool hit = (filter_ctx->use_extra_info_ && OB_NOT_NULL(extra_info))
                         ? filter_ctx->filter_->test(extra_info)
                         : filter_ctx->filter_->test(id);
    keep = filter_ctx->reverse_ ? hit : !hit;
  }
  return keep;
}

int hipvs_search(HipvsHandle *handle, float *query_vector, int64_t topk,
                 const float *&result_dist, const int64_t *&result_ids,
                 const char *&extra_info, int64_t &result_size, int ef_search,
                 void *invalid, bool reverse_filter, bool is_extra_info_filter,
                 float valid_ratio, void *allocator, bool need_extra_info,
                 float distance_threshold, std::unordered_set<int64_t> *returned)
{
  int ret = OB_SUCCESS;
  obhipvs::SearchOptions options;
  options.topk = topk;
  options.ef_search = ef_search;
  options.need_extra_info = need_extra_info;
  options.search_allocator = allocator;
  options.distance_threshold = distance_threshold;
  options.already_returned =
      OB_NOT_NULL(returned) ? static_cast<int64_t>(returned->size()) : 0;

  HipvsFilterCtx filter_ctx;
  filter_ctx.filter_ = static_cast<obvsag::FilterInterface *>(invalid);
  filter_ctx.use_extra_info_ = is_extra_info_filter;
  filter_ctx.reverse_ = reverse_filter;
  filter_ctx.returned_ = returned;
  if (OB_NOT_NULL(filter_ctx.filter_) || OB_NOT_NULL(returned)) {
    options.filter.accept = hipvs_filter_accept;
    options.filter.ctx = &filter_ctx;
    options.filter.valid_ratio = valid_ratio;
  }

  obhipvs::SearchResult result;
  if (OB_FAIL(obhipvs::search(handle->index_, query_vector, options, result))) {
  } else {
    result_dist = result.distances;
    result_ids = result.ids;
    extra_info = result.extra_infos;
    result_size = result.count;
    if (OB_NOT_NULL(returned)) {
      for (int64_t i = 0; i < result.count; ++i) {
        returned->insert(result.ids[i]);
      }
    }
  }
  return ret;
}

}  // namespace

void ObVsagLogger::SetLevel(Level Log_level)
{
}

void ObVsagLogger::Trace(const std::string& msg)
{
    ObString Log = ObString(msg.size(), msg.c_str());
}

void ObVsagLogger::Debug(const std::string& msg)
{
    ObString Log = ObString(msg.size(), msg.c_str());
}

void ObVsagLogger::Info(const std::string& msg)
{
    ObString Log = ObString(msg.size(), msg.c_str());
}

void ObVsagLogger::Warn(const std::string& msg)
{   
    ObString Log = ObString(msg.size(), msg.c_str());
    LOG_WARN_RET(OB_ERR_VSAG_RETURN_ERROR, "[Vsag]",K(Log));
}

void ObVsagLogger::Error(const std::string& msg)
{
    ObString Log = ObString(msg.size(), msg.c_str());
    LOG_ERROR_RET(OB_ERR_VSAG_RETURN_ERROR, "[Vsag]",K(Log));
}

void ObVsagLogger::Critical(const std::string& msg)
{
    ObString Log = ObString(msg.size(), msg.c_str());
    LOG_ERROR_RET(OB_ERR_VSAG_RETURN_ERROR, "[Vsag]",K(Log));
}

int init_vasg_logger(void* logger)
{
    if (!check_vsag_init()) {
        return -4016;
    } else {
        obvsag::set_logger(logger);
        obvsag::set_log_level(static_cast<vsag::Logger::Level>(OB_LOGGER.get_log_level()));
    }
    return 0;
}


bool check_vsag_init()
{
    return obvsag::is_init();
}


int create_index(obvsag::VectorIndexPtr& index_handler, int index_type,
                 const char* dtype, const char* metric, int dim,
                 int max_degree, int ef_construction, int ef_search,
                 void* allocator, int extra_info_size /*= 0*/,
                 int16_t refine_type /*= 0*/, int16_t bq_bits_query /*= 32*/,
                 bool bq_use_fht /*= false*/, int lib /*= obhipvs::LIB_VSAG*/)
{
  int ret = OB_SUCCESS;
  LOG_INFO("vector index create params: ", K(index_type), K(dim), KCSTRING(dtype), KCSTRING(metric), K(max_degree), K(ef_construction), K(ef_search),
      KP(allocator), K(extra_info_size), K(refine_type), K(bq_bits_query), K(bq_use_fht), K(lib));
  if (lib == obhipvs::LIB_HIPVS) {
    if (OB_NOT_NULL(index_handler)) {
      ret = OB_INVALID_ARGUMENT;
      LOG_WARN("hipvs create index expects an empty handler", K(ret), KP(index_handler));
    } else if (!obhipvs::supports(index_type, dtype, metric)) {
      ret = OB_NOT_SUPPORTED;
      LOG_WARN("hipvs backend only supports float32 hnsw", K(ret), K(index_type),
               KCSTRING(dtype), KCSTRING(metric));
    } else {
      HipvsHandle *handle = new (std::nothrow) HipvsHandle();
      if (OB_ISNULL(handle)) {
        ret = OB_ALLOCATE_MEMORY_FAILED;
      } else {
        handle->index_ = nullptr;
        handle->config_.dim = dim;
        handle->config_.metric = metric;
        handle->config_.graph_degree = max_degree;
        handle->config_.intermediate_graph_degree = 0;
        handle->config_.ef_construction = ef_construction;
        handle->config_.ef_search = ef_search;
        handle->config_.extra_info_size = extra_info_size;
        handle->config_.allocator = allocator;
        if (OB_FAIL(obhipvs::create(handle->index_, handle->config_))) {
          LOG_WARN("create hipvs index failed", K(ret));
          delete handle;
          handle = nullptr;
        } else {
          index_handler = static_cast<obvsag::VectorIndexPtr>(handle);
          register_hipvs_handle(index_handler, handle);
        }
      }
    }
  } else {
    obvsag::set_block_size_limit(2*1024*1024);
    ret = obvsag::create_index(index_handler,
                               static_cast<obvsag::IndexType>(index_type),
                               dtype, metric,
                               dim,
                               max_degree,
                               ef_construction,
                               ef_search,
                               allocator,
                               extra_info_size,
                               refine_type,
                               bq_bits_query,
                               bq_use_fht);
  }
  return ret;
}

int validate_create_index(const CreateIndexParam &param, std::string &err_msg)
{
  obvsag::set_block_size_limit(2*1024*1024);
  LOG_INFO("vector index validate params: ", K(param.index_type_), KCSTRING(param.dtype_),
      KCSTRING(param.metric_), K(param.is_sparse_), K(param.dim_), K(param.max_degree_),
      K(param.ef_construction_), K(param.ef_search_), K(param.extra_info_size_),
      K(param.refine_type_), K(param.bq_bits_query_), K(param.bq_use_fht_),
      K(param.use_reorder_), K(param.doc_prune_ratio_), K(param.window_size_), KP(param.allocator_));
  return obvsag::validate_create_index(param, err_msg);
}

int create_index(obvsag::VectorIndexPtr &index_handler, int index_type, const char *dtype, const char *metric,
    bool use_reorder, float doc_prune_ratio, int window_size, void *allocator, int extra_info_size /* 0 */,
    int lib /* = obhipvs::LIB_VSAG */)
{
  int ret = OB_SUCCESS;
  LOG_INFO("vector index create params: ", K(index_type), KCSTRING(dtype), KCSTRING(metric), K(use_reorder), K(doc_prune_ratio), K(window_size), KP(allocator), K(extra_info_size), K(lib));
  if (lib == obhipvs::LIB_HIPVS) {
    // Sparse and IVF indexes have no CAGRA counterpart.
    ret = OB_NOT_SUPPORTED;
    LOG_WARN("hipvs backend does not support sparse or ivf indexes", K(ret), K(index_type));
  } else {
    obvsag::set_block_size_limit(2*1024*1024);
    ret = obvsag::create_index(index_handler, static_cast<obvsag::IndexType>(index_type),
                               dtype, metric, use_reorder, doc_prune_ratio, window_size,
                               allocator, extra_info_size);
  }
  return ret;
}

int build_index(obvsag::VectorIndexPtr index_handler, float* vector_list, int64_t* ids, int dim, int size, char* extra_info /*= nullptr*/)
{
  int ret = OB_SUCCESS;
  HipvsHandle *handle = find_hipvs_handle(index_handler);
  if (OB_NOT_NULL(handle)) {
    ret = obhipvs::build(handle->index_, vector_list, ids, size, extra_info);
  } else {
    ret = obvsag::build_index(index_handler, vector_list, ids, dim, size, extra_info);
  }
  return ret;
}

int build_index(obvsag::VectorIndexPtr &index_handler, uint32_t *lens, uint32_t *dims, float *vals, int64_t *ids,
    int size, char *extra_infos /*= nullptr*/)
{
  int ret = OB_SUCCESS;
  if (OB_NOT_NULL(find_hipvs_handle(index_handler))) {
    ret = OB_NOT_SUPPORTED;
    LOG_WARN("hipvs backend does not support sparse build", K(ret));
  } else {
    ret = obvsag::build_index(index_handler, lens, dims, vals, ids, size, extra_infos);
  }
  return ret;
}

int add_index(obvsag::VectorIndexPtr index_handler, float* vector_list, int64_t* ids, int dim, char *extra_info, int size)
{
  int ret = OB_SUCCESS;
  HipvsHandle *handle = find_hipvs_handle(index_handler);
  if (OB_NOT_NULL(handle)) {
    ret = obhipvs::extend(handle->index_, vector_list, ids, size, extra_info);
  } else {
    ret = obvsag::add_index(index_handler, vector_list, ids, dim, size, extra_info);
  }
  return ret;
}

int add_index(obvsag::VectorIndexPtr &index_handler, uint32_t *lens, uint32_t *dims, float *vals, int64_t *ids, int size,
    char *extra_infos)
{
  int ret = OB_SUCCESS;
  if (OB_NOT_NULL(find_hipvs_handle(index_handler))) {
    ret = OB_NOT_SUPPORTED;
    LOG_WARN("hipvs backend does not support sparse add", K(ret));
  } else {
    ret = obvsag::add_index(index_handler, lens, dims, vals, ids, size, extra_infos);
  }
  return ret;
}

int get_index_number(obvsag::VectorIndexPtr index_handler, int64_t &size)
{
  int ret = OB_SUCCESS;
  HipvsHandle *handle = find_hipvs_handle(index_handler);
  if (OB_NOT_NULL(handle)) {
    ret = obhipvs::get_count(handle->index_, size);
  } else {
    ret = obvsag::get_index_number(index_handler, size);
  }
  return ret;
}

int get_index_type(obvsag::VectorIndexPtr index_handler)
{
  int index_type = obvsag::INVALID_INDEX_TYPE;
  HipvsHandle *handle = find_hipvs_handle(index_handler);
  if (OB_NOT_NULL(handle)) {
    // CAGRA replaces only plain HNSW, the query layer keys off this type.
    index_type = obvsag::HNSW_TYPE;
  } else {
    index_type = obvsag::get_index_type(index_handler);
  }
  return index_type;
}

int cal_distance_by_id(obvsag::VectorIndexPtr index_handler,
                       const float *vector,
                       const int64_t *ids,
                       int64_t count,
                       const float *&distances)
{
  int ret = OB_SUCCESS;
  HipvsHandle *handle = find_hipvs_handle(index_handler);
  if (OB_NOT_NULL(handle)) {
    ret = obhipvs::calc_distance_by_id(handle->index_, vector, ids, count, distances);
  } else {
    ret = obvsag::cal_distance_by_id(index_handler, vector, ids, count, distances);
  }
  return ret;
}

int cal_distance_by_id(obvsag::VectorIndexPtr index_handler,
                       uint32_t len, uint32_t *dims, float *vals,
                       const int64_t *ids,
                       int64_t count,
                       const float *&distances)
{
  int ret = OB_SUCCESS;
  if (OB_NOT_NULL(find_hipvs_handle(index_handler))) {
    ret = OB_NOT_SUPPORTED;
    LOG_WARN("hipvs backend does not support sparse distance by id", K(ret));
  } else {
    ret = obvsag::cal_distance_by_id(index_handler, len, dims, vals, ids, count, distances);
  }
  return ret;
}

int get_vid_bound(obvsag::VectorIndexPtr index_handler, int64_t &min_vid, int64_t &max_vid)
{
  int ret = OB_SUCCESS;
  HipvsHandle *handle = find_hipvs_handle(index_handler);
  if (OB_NOT_NULL(handle)) {
    ret = obhipvs::get_vid_bound(handle->index_, min_vid, max_vid);
  } else {
    ret = obvsag::get_vid_bound(index_handler, min_vid, max_vid);
  }
  return ret;
}

int get_extra_info_by_ids(obvsag::VectorIndexPtr& index_handler, 
                          const int64_t* ids, 
                          int64_t count, 
                          char *extra_infos) {
  int ret = OB_SUCCESS;
  HipvsHandle *handle = find_hipvs_handle(index_handler);
  if (OB_NOT_NULL(handle)) {
    ret = obhipvs::get_extra_info_by_ids(handle->index_, ids, count, extra_infos);
  } else {
    ret = obvsag::get_extra_info_by_ids(index_handler, ids, count, extra_infos);
  }
  return ret;
}

int knn_search(obvsag::VectorIndexPtr index_handler, float* query_vector,int dim, int64_t topk,
               const float*& result_dist, const int64_t*& result_ids, const char *&extra_info, int64_t &result_size, int ef_search,
               void* invalid, bool reverse_filter, bool is_extra_info_filter, float valid_ratio, void *allocator, bool need_extra_info,
               float distance_threshold)
{
  int ret = OB_SUCCESS;
  HipvsHandle *handle = find_hipvs_handle(index_handler);
  if (OB_NOT_NULL(handle)) {
    ret = hipvs_search(handle, query_vector, topk, result_dist, result_ids, extra_info,
                       result_size, ef_search, invalid, reverse_filter,
                       is_extra_info_filter, valid_ratio, allocator, need_extra_info,
                       distance_threshold, nullptr);
  } else {
    ret = obvsag::knn_search(index_handler, query_vector, dim, topk,
                             result_dist, result_ids, result_size,
                             ef_search, need_extra_info, extra_info,
                             invalid, reverse_filter, is_extra_info_filter,
                             allocator, valid_ratio, distance_threshold);
  }
  return ret;
}

int knn_search(obvsag::VectorIndexPtr index_handler, float* query_vector,int dim, int64_t topk,
               const float*& result_dist, const int64_t*& result_ids, const char *&extra_info, int64_t &result_size, int ef_search,
               void* invalid, bool reverse_filter, bool is_extra_info_filter, float valid_ratio, void *allocator,
               bool need_extra_info, void *&iter_ctx, bool is_last_search)
{
  int ret = OB_SUCCESS;
  HipvsHandle *handle = find_hipvs_handle(index_handler);
  if (OB_ISNULL(handle)) {
    ret = obvsag::knn_search(index_handler, query_vector, dim, topk,
                             result_dist, result_ids, result_size,
                             ef_search, need_extra_info, extra_info,
                             invalid, reverse_filter, is_extra_info_filter,
                             valid_ratio, iter_ctx, is_last_search, allocator);
  } else {
    HipvsIterator *iter = nullptr;
    bool fresh_iter = false;
    if (OB_NOT_NULL(iter_ctx)) {
      iter = find_hipvs_iterator(iter_ctx);
      if (OB_ISNULL(iter)) {
        ret = OB_INVALID_ARGUMENT;
        LOG_WARN("hipvs iterator context does not belong to the hipvs backend", K(ret), KP(iter_ctx));
      }
    } else {
      iter = new (std::nothrow) HipvsIterator();
      if (OB_ISNULL(iter)) {
        ret = OB_ALLOCATE_MEMORY_FAILED;
      } else {
        register_hipvs_iterator(iter);
        iter_ctx = static_cast<void *>(iter);
        fresh_iter = true;
      }
    }
    if (OB_SUCC(ret)) {
      ret = hipvs_search(handle, query_vector, topk, result_dist, result_ids, extra_info,
                         result_size, ef_search, invalid, reverse_filter,
                         is_extra_info_filter, valid_ratio, allocator, need_extra_info,
                         FLT_MAX, &iter->returned_);
      if (OB_FAIL(ret) && fresh_iter) {
        take_hipvs_iterator(iter_ctx);
        delete iter;
        iter_ctx = nullptr;
      }
    }
  }
  return ret;
}

int knn_search(obvsag::VectorIndexPtr index_handler, uint32_t len, uint32_t *dims, float *vals, int64_t topk,
    const float *&result_dist, const int64_t *&result_ids, const char *&extra_info, int64_t &result_size, float query_prune_ratio, int64_t n_candidate,
    void *invalid, bool reverse_filter,
    bool is_extra_info_filter, float valid_ratio, void *allocator, bool need_extra_info)
{
  int ret = OB_SUCCESS;
  if (OB_NOT_NULL(find_hipvs_handle(index_handler))) {
    ret = OB_NOT_SUPPORTED;
    LOG_WARN("hipvs backend does not support sparse search", K(ret));
  } else {
    ret = obvsag::knn_search(index_handler, len, dims, vals, topk,
                             result_dist, result_ids, extra_info, result_size,
                             query_prune_ratio, n_candidate,
                             invalid, reverse_filter, is_extra_info_filter,
                             valid_ratio, allocator, need_extra_info);
  }
  return ret;
}

int fserialize(obvsag::VectorIndexPtr index_handler, std::ostream& out_stream)
{
  int ret = OB_SUCCESS;
  HipvsHandle *handle = find_hipvs_handle(index_handler);
  if (OB_NOT_NULL(handle)) {
    ret = obhipvs::serialize(handle->index_, out_stream);
  } else {
    ret = obvsag::fserialize(index_handler, out_stream);
  }
  return ret;
}

int fdeserialize(obvsag::VectorIndexPtr& index_handler, std::istream& in_stream)
{
  int ret = OB_SUCCESS;
  HipvsHandle *handle = find_hipvs_handle(index_handler);
  if (OB_NOT_NULL(handle)) {
    ret = obhipvs::deserialize(handle->index_, in_stream);
  } else {
    ret = obvsag::fdeserialize(index_handler, in_stream);
  }
  return ret;
}

int delete_index(obvsag::VectorIndexPtr& index_handler)
{
  int ret = OB_SUCCESS;
  HipvsHandle *handle = take_hipvs_handle(index_handler);
  if (OB_NOT_NULL(handle)) {
    obhipvs::destroy(handle->index_);
    delete handle;
    handle = nullptr;
    index_handler = nullptr;
  } else {
    ret = obvsag::delete_index(index_handler);
  }
  return ret;
}

void delete_iter_ctx(void *iter_ctx)
{
  HipvsIterator *iter = take_hipvs_iterator(iter_ctx);
  if (OB_NOT_NULL(iter)) {
    delete iter;
    iter = nullptr;
  } else {
    obvsag::delete_iter_ctx(iter_ctx);
  }
}

// return byte
uint64_t estimate_memory(obvsag::VectorIndexPtr& index_handler, const uint64_t row_count, const bool is_build)
{
  uint64_t estimate = 0;
  HipvsHandle *handle = find_hipvs_handle(index_handler);
  if (OB_NOT_NULL(handle)) {
    estimate = obhipvs::estimate_memory(handle->config_, static_cast<int64_t>(row_count), is_build);
  } else {
    estimate = obvsag::estimate_memory(index_handler, row_count, is_build);
  }
  return estimate;
}

int immutable_optimize(obvsag::VectorIndexPtr& index_handler)
{
  int ret = OB_SUCCESS;
  HipvsHandle *handle = find_hipvs_handle(index_handler);
  if (OB_NOT_NULL(handle)) {
    // The DDL path only appends rows, the CAGRA graph is built here once the
    // snapshot is sealed (or when a snapshot without a graph is loaded).
    ret = obhipvs::optimize(handle->index_);
  } else {
    ret = obvsag::immutable_optimize(index_handler);
  }
  return ret;
}

} //namespace obvectorlib
} //namespace common
} //namespace oceanbase
