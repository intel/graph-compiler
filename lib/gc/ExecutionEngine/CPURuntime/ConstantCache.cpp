#include "gc/ExecutionEngine/CPURuntime/ConstantCache.hpp"
#include <string.h>
#include <unordered_map>

namespace mlir::gc {

ConstCacheProxy::~ConstCacheProxy() = default;


CachedGraphTensor::CachedGraphTensor(
    const std::shared_ptr<ConstCacheProxy> &base, size_t offset)
    : base{base}, offset{offset} {
  // todo: fill in real values
  ref.basePtr = (char *)base->get_buffer_unsafe() + offset;
  ref.data = ref.basePtr;
  ref.offset = 0;
  memset(ref.sizes, 0, sizeof(ref.sizes));
  memset(ref.strides, 0, sizeof(ref.strides));
}

static std::unordered_map<uint64_t, std::shared_ptr<CachedGraphTensor>> cache;

std::shared_ptr<CachedGraphTensor> queryCacheTensor(uint64_t key) {
  auto itr = cache.find(key);
  if (itr != cache.end()) {
    return itr->second;
  }
  return nullptr;
}

bool regCachedTensor(uint64_t key,
                       const std::shared_ptr<ConstCacheProxy> &base,
                       size_t offset) {
  if (queryCacheTensor(key)) {
    return false;
  }
  cache[key] = std::make_shared<CachedGraphTensor>(base, offset);
  return true;
}
} // namespace mlir::gc