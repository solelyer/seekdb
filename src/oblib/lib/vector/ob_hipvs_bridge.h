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

#pragma once

#include <cfloat>
#include <cstdint>
#include <iosfwd>

namespace oceanbase {
namespace common {
namespace obhipvs {

// The values must stay in sync with ObVectorIndexAlgorithmLib. The query layer
// asserts this in ob_vector_index_util.cpp.
enum LibType {
  LIB_VSAG = 0,
  LIB_OB = 1,
  LIB_HIPVS = 2,
};

struct Capability {
  // The bridge was compiled with hipVS support (-DOB_USE_HIPVS=ON).
  bool compiled = false;
  // A usable device and the cuVS runtime were found at runtime.
  bool available = false;
  int32_t device_count = 0;
  int32_t runtime_version = 0;
  char device_name[128] = {0};
  char message[256] = {0};
};

// Probed once per process, the result is cached. Thread safe.
const Capability &capability();

struct Config {
  int32_t dim = 0;
  // "l2", "ip" or "cosine", the same strings seekdb passes to vsag.
  const char *metric = nullptr;
  // HNSW M, used as the CAGRA graph degree.
  int32_t graph_degree = 0;
  int32_t intermediate_graph_degree = 0;
  int32_t ef_construction = 0;
  int32_t ef_search = 0;
  int32_t extra_info_size = 0;
  // vsag::Allocator used for CPU side metadata and cal_distance_by_id results.
  void *allocator = nullptr;
};

// True when the requested index shape can be served by the GPU backend. Cheap,
// does not touch the device.
bool supports(int32_t index_type, const char *dtype, const char *metric);

class CagraIndex;

int create(CagraIndex *&index, const Config &config);
void destroy(CagraIndex *&index);

int build(CagraIndex *index, const float *vectors, const int64_t *ids, int64_t count,
          const char *extra_infos);
// Append rows. When the graph is already built it is extended on the GPU and a
// failed extension falls back to a full rebuild; otherwise rows are only staged
// on the host until optimize() or serialize() builds the graph.
int extend(CagraIndex *index, const float *vectors, const int64_t *ids, int64_t count,
           const char *extra_infos);
// Builds the CAGRA graph once rows have been staged, no-op when the graph is
// already built or the index is empty.
int optimize(CagraIndex *index);

struct SearchFilter {
  // Returns true when the candidate may be returned. Either id or extra_info is
  // examined by the caller, mirroring obvsag::FilterInterface.
  bool (*accept)(void *ctx, int64_t id, const char *extra_info) = nullptr;
  void *ctx = nullptr;
  // Fraction of the rows expected to pass the filter, used to over-fetch.
  float valid_ratio = 1.0f;
};

struct SearchOptions {
  int64_t topk = 0;
  int32_t ef_search = 0;
  bool need_extra_info = false;
  // Rows handed out by earlier pages of an iterative search. The search window
  // is widened by this amount so that paging does not stall once the first
  // over-fetched candidates are exhausted.
  int64_t already_returned = 0;
  // vsag::Allocator that owns the returned arrays, may be nullptr.
  void *search_allocator = nullptr;
  SearchFilter filter;
  float distance_threshold = FLT_MAX;
};

struct SearchResult {
  const float *distances = nullptr;
  const int64_t *ids = nullptr;
  const char *extra_infos = nullptr;
  int64_t count = 0;
};

int search(CagraIndex *index, const float *query, const SearchOptions &options,
           SearchResult &result);

// Allocated with Config::allocator so that the adaptor releases it through the
// index memory context, exactly like the vsag implementation.
int calc_distance_by_id(CagraIndex *index, const float *query, const int64_t *ids,
                        int64_t count, const float *&distances);
int get_extra_info_by_ids(CagraIndex *index, const int64_t *ids, int64_t count,
                          char *extra_infos);
int get_count(CagraIndex *index, int64_t &count);
int get_vid_bound(CagraIndex *index, int64_t &min_vid, int64_t &max_vid);
uint64_t estimate_memory(const Config &config, int64_t count, bool is_build);

// The on-disk payload is a self describing container so that a hipVS index and
// a CPU index that had to fall back can coexist.
int serialize(CagraIndex *index, std::ostream &out);
int deserialize(CagraIndex *&index, std::istream &in);

}  // namespace obhipvs
}  // namespace common
}  // namespace oceanbase
