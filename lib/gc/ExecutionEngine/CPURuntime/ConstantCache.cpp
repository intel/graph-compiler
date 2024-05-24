#include "gc/ExecutionEngine/CPURuntime/ConstantCache.hpp"
#include <string.h>
#include <unordered_map>

namespace mlir::gc {

const_cache_proxy::~const_cache_proxy() = default;


cached_graph_tensor::cached_graph_tensor(
    const std::shared_ptr<const_cache_proxy> &base, size_t offset)
    : base{base}, offset{offset} {
  // todo: fill in real values
  ref.basePtr = (char *)base->get_buffer_unsafe() + offset;
  ref.data = ref.basePtr;
  ref.offset = 0;
  memset(ref.sizes, 0, sizeof(ref.sizes));
  memset(ref.strides, 0, sizeof(ref.strides));
}

static std::unordered_map<uint64_t, std::shared_ptr<cached_graph_tensor>> cache;

std::shared_ptr<cached_graph_tensor> query_cached_tensor(uint64_t key) {
  auto itr = cache.find(key);
  if (itr != cache.end()) {
    return itr->second;
  }
  return nullptr;
}

bool reg_cached_tensor(uint64_t key,
                       const std::shared_ptr<const_cache_proxy> &base,
                       size_t offset) {
  if (query_cached_tensor(key)) {
    return false;
  }
  cache[key] = std::make_shared<cached_graph_tensor>(base, offset);
  return true;
}
} // namespace mlir::gc