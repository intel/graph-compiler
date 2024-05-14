#ifndef GC_VERSION_H
#define GC_VERSION_H

#if !defined(GC_VERSION_MAJOR) || !defined(GC_VERSION_MINOR) || !defined(GC_VERSION_PATCH)
// define an invalid version if it wasn't defined by CMake
#include <limits>
#define GC_VERSION_MAJOR std::numeric_limits<uint8_t>::max()
#define GC_VERSION_MINOR std::numeric_limits<uint8_t>::max()
#define GC_VERSION_PATCH std::numeric_limits<uint8_t>::max()
#endif
#ifndef GC_VERSION_HASH
#define GC_VERSION_HASH "N/A"
#endif

#endif
