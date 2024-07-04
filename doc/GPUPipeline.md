# GPU Pipeline overview

This is a living document for GPU pipeline design. It's purpose is to keep the decision history and provide a guiding overview for development. We expect swift changes in the design as we go, so this mostly highlights guiding principles.

## Initial state description

The primary goal of the design is to adhere to certain qualities of the final solution.
The spirit of the design is to reuse the existing parts, prefer upstream, and target long-term support in conjunction with other devices.

At the highest level, the pipeline can be split into three main stages:
1. High-level platform-independent* transformations. These are to be shared with other flows (e.g., fusion).
2. GPU-specific transformations. These are responsible for HW mapping and include everything until a SPIR-V is emitted.
3. Code generation. This is tailored to a particular platform and is performed by a backend.

There are existing paths for each stage (sometimes multiple, the choice affects other parts). A short landscape description follows.

### Landscape
There are two primary ways of generating GPU target binary code, both going through IGC: scalar and vector paths.

The scalar (aka SIMT) path relies on IGC's vectorization capabilities to map logical threads to SIMD lanes. Handling synchronization (e.g., cross-lane communication) is the main burden for otherwise transformation-amenable representation. 

The vector (aka SIMD) path in IGC expects the IR to have a certain explicitly-vectorized form, primarily built via a set of intrinsics (VC-intinsics). The main complexity of the approach for the pipeline is handling data/compute distribution between those vectors and handling such a deviation from other GPU types lowering paths.

Today, there are two main options to reach the low-level compiler:
1. Lower to SPIR-V dialect and serialize it (IMEX).
2. Lower to LLVM IR and use the SPIR-V Translator (Triton).

Both produce a SPIR-V that can be consumed by IGC.

Going up the pipeline, the abstractions needed to express specific ISA semantics (e.g., DPAS and nd-load required for efficient contraction implementation) are covered by XeGPU dialect. The dialect allows for both SIMT and SIMD -style lowering.

TODO: gpu(x), linalg-to-scf, gpu-map-parallel-loops.

### The path of least resistance
First milestone for the pipeline creation aims at taking what's working now and putting it together.

This includes:
- Going through XeGPU dialect
- Using IMEX's XeGPU lowering
- Adapting TPP's linalg-to-xegpu
