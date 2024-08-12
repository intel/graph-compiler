#include "umf/pools/pool_scalable.h"
#include "umf/providers/provider_os_memory.h"
#include <umf.h>

#include <stdio.h>
#include <string.h>

namespace {

static constexpr size_t defaultAlignment = 64;

umf_memory_provider_ops_t *provider_ops = umfOsMemoryProviderOps();
umf_os_memory_provider_params_t params = umfOsMemoryProviderParamsDefault();
umf_memory_provider_handle_t provider;
umf_result_t res_provider =
    umfMemoryProviderCreate(provider_ops, &params, &provider);

umf_memory_pool_handle_t pool;

umf_result_t res_pool =
    umfPoolCreate(umfScalablePoolOps(), provider, NULL, 0, &pool);

} // namespace

extern "C" void *umfAlignedMalloc(size_t sz) noexcept {
  return umfPoolAlignedMalloc(pool, sz, defaultAlignment);
}

extern "C" void umfAlignedFree(void *p) noexcept { umfFree(p); }
