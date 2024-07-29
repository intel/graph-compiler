import ctypes
import numpy as np
import os

# Load libnuma
libnuma = ctypes.CDLL("libnuma.so.1")

# Define numa_alloc_onnode function
libnuma.numa_alloc_onnode.restype = ctypes.c_void_p
libnuma.numa_alloc_onnode.argtypes = [ctypes.c_size_t, ctypes.c_int]

# Define numa_free function
libnuma.numa_free.argtypes = [ctypes.c_void_p, ctypes.c_size_t]

def allocate_memory_on_node(size, numa_node):
    # Allocate memory on the specified NUMA node
    buffer_addr = libnuma.numa_alloc_onnode(size, numa_node)
    if not buffer_addr:
        raise MemoryError(f"Failed to allocate memory on NUMA node {numa_node}")
    
    return buffer_addr

# Example usage:
if __name__ == "__main__":
    # Allocate memory on NUMA node 0
    size = 1024 * 1024 * 100  # 100 MB
    numa_node = 0
    buffer_addr = allocate_memory_on_node(size, numa_node)
    print(f"Allocated memory on NUMA node {numa_node}: {buffer_addr}")

    # Read and print numa_maps to verify the allocation
    pid = os.getpid()
    with open(f"/proc/{pid}/numa_maps", "r") as f:
        for line in f:
            print(line.strip())
