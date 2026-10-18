//===-- GpuToGcGpu.cpp - GPU ops to GC GPU runtime calls --------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <optional>

#include <llvm/ADT/DenseMap.h>

#include <mlir/Conversion/LLVMCommon/ConversionTarget.h>
#include <mlir/Conversion/LLVMCommon/Pattern.h>
#include <mlir/Conversion/Passes.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dominance.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include "gc/GpuCompiler.h"
#include "gc/Runtime/GpuRuntime.h"
#include "gc/Utils/Transform.h"

using namespace mlir;
// The runtime function names, see gc/Runtime/GpuRuntime.h.
using namespace ::gc::gpu;

namespace mlir::gc {
#define GEN_PASS_DECL_GPUTOGCGPU
#define GEN_PASS_DEF_GPUTOGCGPU
#include "gc/Transforms/Passes.h.inc"
} // namespace mlir::gc

namespace {
// The generated per-kernel globals and functions are named
// gcGpuKernel_<kernel>_{Ptr,Name,Bin,ArgTypes}, gcGpuGetKernel_<kernel>
// and gcGpuCreateKernel_<kernel>.
constexpr char KERNEL_GLOBAL[] = "gcGpuKernel_";
constexpr char KERNEL_GETTER[] = "gcGpuGetKernel_";
constexpr char KERNEL_CREATOR[] = "gcGpuCreateKernel_";

namespace RtSym = ::gc::gpu::sym;
using ProgSym = Program::Symbols;

StringRef callee(Operation *op) {
  auto call = dyn_cast<LLVM::CallOp>(op);
  return call ? call.getCallee().value_or(StringRef()) : StringRef();
}

bool isCall(Operation *op, StringRef name) { return callee(op) == name; }

// The MLIR GPU runtime (mgpu*) functions are replaced by this pass.
bool isMgpu(StringRef name) { return name.starts_with("mgpu"); }

bool isMgpuCall(Operation *op) { return isMgpu(callee(op)); }

bool isPtr(Value value) { return isa<LLVM::LLVMPointerType>(value.getType()); }

// A gcGpuLaunch or gcGpuMemcpy call and the events it depends on.
using CallDeps = std::pair<LLVM::CallOp, SmallVector<Value>>;

// Calls the function, declaring it on the first use. The signature is derived
// from the argument types.
LLVM::CallOp funcCall(OpBuilder &builder, const StringRef name,
                      const Type returnType, const Location loc,
                      const ValueRange arguments) {
  auto module = builder.getBlock()->getParent()->getParentOfType<ModuleOp>();
  auto function = module.lookupSymbol<LLVM::LLVMFuncOp>(name);
  if (!function) {
    auto type = LLVM::LLVMFunctionType::get(
        returnType, llvm::to_vector(TypeRange(arguments)));
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(module.getBody());
    function = LLVM::LLVMFuncOp::create(builder, loc, name, type);
  }
  return LLVM::CallOp::create(builder, loc, function, arguments);
}

// Creates a public function without arguments at the end of the module and
// sets the insertion point to its entry block.
void createPublicFunc(OpBuilder &rewriter, ModuleOp mod, StringRef name,
                      Type returnType) {
  rewriter.setInsertionPointToEnd(mod.getBody());
  auto func = LLVM::LLVMFuncOp::create(
      rewriter, mod.getLoc(), name, LLVM::LLVMFunctionType::get(returnType, {}),
      LLVM::Linkage::External);
  rewriter.setInsertionPointToStart(func.addEntryBlock(rewriter));
}

struct Helper final {
  LLVMTypeConverter &converter;
  Type voidType;
  Type ptrType;
  Type i32Type;
  // size_t
  Type sizeType;
  // sizeof(void*)
  unsigned ptrSize;
  // The runtime argument, appended to the entry function.
  Value runtime;
  // The external dependencies, appended to the entry function. Every kernel
  // launch and copy waits for these events, that are owned by the caller.
  Value deps;
  Value nDeps;
  // The kernel pointer globals, in the launch order.
  mutable SmallVector<LLVM::GlobalOp> kernels;
  // The pointer arguments of each gcGpuLaunch call. Used to find the
  // dependencies between the kernels, the allocations and the copies.
  mutable DenseMap<Operation *, SmallVector<Value>> kernelPtrArgs;
  // The shared stack buffers, allocated in the entry block on the first use and
  // reused by all the runtime calls: `ptrArr` contains the pointer arrays (the
  // kernel argument pointers or the event lists) and `valArr` - the kernel
  // argument values. Reusing is safe, because the runtime reads the arrays
  // during the call. Both consist of ptr-sized slots, the number of which is
  // only known after all the calls are created, thus the size is set by
  // finalize().
  Block *entry;
  mutable Value ptrArr;
  mutable Value valArr;
  mutable unsigned ptrSlots = 0;
  mutable unsigned valSlots = 0;

  Helper(MLIRContext *ctx, LLVMTypeConverter &converter, Block *entry,
         Value runtime, Value deps, Value nDeps)
      : converter(converter), voidType(LLVM::LLVMVoidType::get(ctx)),
        ptrType(LLVM::LLVMPointerType::get(ctx)),
        i32Type(IntegerType::get(ctx, 32)),
        sizeType(IntegerType::get(ctx, converter.getPointerBitwidth())),
        ptrSize(converter.getPointerBitwidth() / 8), runtime(runtime),
        deps(deps), nDeps(nDeps), entry(entry) {}

  // Sets the actual size of the shared buffers.
  void finalize() const {
    for (auto [buf, slots] :
         {std::pair{ptrArr, ptrSlots}, std::pair{valArr, valSlots}}) {
      if (!buf) continue;
      auto *arr = buf.getDefiningOp();
      OpBuilder rewriter(arr);
      auto *size = arr->getOperand(0).getDefiningOp();
      arr->setOperand(0, constant(rewriter, arr->getLoc(), sizeType, slots));
      if (size->use_empty()) size->erase();
    }
  }

  Value constant(OpBuilder &rewriter, Location loc, Type type,
                 int64_t value) const {
    return LLVM::ConstantOp::create(rewriter, loc, type,
                                    rewriter.getIntegerAttr(type, value));
  }

  Value nullPtr(OpBuilder &rewriter, Location loc) const {
    return LLVM::ZeroOp::create(rewriter, loc, ptrType);
  }

  // The address of the `index` slot of a ptr-sized slots buffer.
  Value slot(OpBuilder &rewriter, Location loc, Value buf,
             unsigned index) const {
    return index
               ? LLVM::GEPOp::create(rewriter, loc, ptrType, ptrType, buf,
                                     LLVM::GEPArg(static_cast<int32_t>(index)))
                     .getResult()
               : buf;
  }

  // Allocates the shared buffer at the beginning of the entry block on the
  // first use. The size is a placeholder, set by finalize().
  Value buffer(OpBuilder &rewriter, Value &buf, unsigned alignment) const {
    if (!buf) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(entry);
      auto loc = entry->getParentOp()->getLoc();
      buf = LLVM::AllocaOp::create(rewriter, loc, ptrType, ptrType,
                                   constant(rewriter, loc, sizeType, 0),
                                   alignment)
                .getResult();
    }
    return buf;
  }

  // Truncates or extends the integer value to the specified type. Returns
  // nullptr if the value is not an integer.
  Value toInt(OpBuilder &rewriter, Location loc, Value value, Type type) const {
    auto from = dyn_cast<IntegerType>(value.getType());
    if (!from) return nullptr;
    auto to = cast<IntegerType>(type);
    if (from == to) return value;
    return from.getWidth() > to.getWidth()
               ? LLVM::TruncOp::create(rewriter, loc, to, value).getResult()
               : LLVM::ZExtOp::create(rewriter, loc, to, value).getResult();
  }

  // Stores the values in the shared pointer buffer and returns its address or
  // null, if the values are empty.
  Value array(OpBuilder &rewriter, Location loc, ValueRange values) const {
    if (values.empty()) return nullPtr(rewriter, loc);
    ptrSlots = std::max<unsigned>(ptrSlots, values.size());
    auto arr = buffer(rewriter, ptrArr, 0);
    for (auto [i, value] : llvm::enumerate(values))
      LLVM::StoreOp::create(rewriter, loc, value, slot(rewriter, loc, arr, i));
    return arr;
  }

  // Reserves `count` slots in the shared value buffer, starting at `index`, and
  // returns the address of the first one.
  Value valueSlot(OpBuilder &rewriter, Location loc, unsigned index,
                  unsigned count) const {
    valSlots = std::max(valSlots, index + count);
    // The values are naturally aligned, up to 16 bytes, e.g. i128.
    return slot(rewriter, loc, buffer(rewriter, valArr, 16), index);
  }

  // The number of the ptr-sized slots, required to store a value of this type.
  unsigned slotCount(Type type) const {
    auto bits = isa<LLVM::LLVMPointerType>(type) ? ptrSize * 8
                                                 : type.getIntOrFloatBitWidth();
    return (bits + ptrSize * 8 - 1) / (ptrSize * 8);
  }

  // gcGpuWait(runtime, events, nEvents). The ownership of the events is
  // transferred to the runtime. The external dependencies are never passed
  // here, they are owned by the caller.
  void wait(OpBuilder &rewriter, Location loc, ValueRange events) const {
    funcCall(rewriter, RtSym::gcGpuWaitName, voidType, loc,
             {runtime, array(rewriter, loc, events),
              constant(rewriter, loc, i32Type, events.size())});
  }

  // Merges the events into the deps array and its size, that are the two last
  // arguments of a gcGpuLaunch or gcGpuMemcpy call and are initialized with the
  // external dependencies. A single array of `nDeps + <the maximum number of
  // events per call>` elements is allocated at the beginning of the block, the
  // external dependencies are copied to its beginning and the events are stored
  // after them before each call. All the calls share this array - the runtime
  // reads the dependencies during the call.
  void setDeps(Block &block, MutableArrayRef<CallDeps> calls) const {
    if (calls.empty()) return;
    size_t max = 0;
    for (auto &[call, events] : calls) max = std::max(max, events.size());

    OpBuilder rewriter(&block, block.begin());
    auto loc = block.getParentOp()->getLoc();
    auto offset = toInt(rewriter, loc, nDeps, sizeType);
    auto arr = LLVM::AllocaOp::create(
        rewriter, loc, ptrType, ptrType,
        LLVM::AddOp::create(rewriter, loc, offset,
                            constant(rewriter, loc, sizeType, max)));
    LLVM::MemcpyOp::create(
        rewriter, loc, arr, deps,
        LLVM::MulOp::create(rewriter, loc, offset,
                            constant(rewriter, loc, sizeType, ptrSize)),
        /*isVolatile=*/false);
    // The event slots, following the external dependencies.
    SmallVector<Value> slots;
    slots.reserve(max);
    for (size_t i = 0; i < max; i++) {
      auto index = i ? LLVM::AddOp::create(rewriter, loc, offset,
                                           constant(rewriter, loc, sizeType, i))
                           .getResult()
                     : offset;
      slots.emplace_back(
          LLVM::GEPOp::create(rewriter, loc, ptrType, ptrType, arr, index));
    }

    for (auto &[call, events] : calls) {
      rewriter.setInsertionPoint(call);
      auto callLoc = call.getLoc();
      for (auto [slot, event] : llvm::zip_first(slots, events))
        LLVM::StoreOp::create(rewriter, callLoc, event, slot);
      auto index = call.getNumOperands() - 2;
      call.setOperand(index, arr);
      call.setOperand(index + 1,
                      LLVM::AddOp::create(
                          rewriter, callLoc, nDeps,
                          constant(rewriter, callLoc, i32Type, events.size())));
    }
  }
};

// The base class of the mgpu* call conversion patterns.
struct ConvertCall : OpRewritePattern<LLVM::CallOp> {
  const Helper &helper;

  ConvertCall(MLIRContext *ctx, const Helper &helper)
      : OpRewritePattern(ctx), helper(helper) {}
};

// Converts gpu.launch_func to a gcGpuLaunch() call. The kernels are created
// lazily and cached in the gcGpuKernel_<kernel>_Ptr global variables.
struct ConvertLaunch final : ConvertOpToLLVMPattern<gpu::LaunchFuncOp> {
  const Helper &helper;

  explicit ConvertLaunch(const Helper &helper)
      : ConvertOpToLLVMPattern(helper.converter), helper(helper) {}

  LogicalResult
  matchAndRewrite(gpu::LaunchFuncOp launch, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = launch.getLoc();
    if (launch->getNumResults()) {
      return launch.emitOpError() << "async launch is not supported";
    }
    auto kernel = getKernel(launch, rewriter);
    if (!kernel) return failure();

    // The runtime expects an array of pointers to the argument values. Both the
    // values and the pointers are stored in the shared buffers, see Helper.
    SmallVector<Value> args;
    SmallVector<Value> ptrArgs;
    unsigned index = 0;
    for (auto arg : adaptor.getKernelOperands()) {
      auto type = arg.getType();
      if (!isPtr(arg) && !type.isIntOrFloat()) {
        return launch.emitOpError()
               << "unsupported kernel argument type " << type
               << ": expected a pointer or a scalar";
      }
      auto count = helper.slotCount(type);
      // Keep the value naturally aligned.
      index = llvm::alignTo(index, count);
      auto slot = helper.valueSlot(rewriter, loc, index, count);
      LLVM::StoreOp::create(rewriter, loc, arg, slot);
      args.emplace_back(slot);
      index += count;
      if (isPtr(arg)) ptrArgs.emplace_back(arg);
    }

    SmallVector<Value> operands{
        helper.runtime, kernel, helper.array(rewriter, loc, args),
        helper.constant(rewriter, loc, helper.i32Type, args.size())};
    for (auto size : {adaptor.getGridSizeX(), adaptor.getGridSizeY(),
                      adaptor.getGridSizeZ(), adaptor.getBlockSizeX(),
                      adaptor.getBlockSizeY(), adaptor.getBlockSizeZ()}) {
      if (auto value = helper.toInt(rewriter, loc, size, helper.i32Type)) {
        operands.emplace_back(value);
      } else {
        return launch.emitOpError() << "the grid and block sizes must be "
                                       "lowered to integer values";
      }
    }
    // The event dependencies are merged into the external ones, if any, after
    // all the operations are converted.
    operands.emplace_back(helper.deps);
    operands.emplace_back(helper.nDeps);

    auto call = funcCall(rewriter, RtSym::gcGpuLaunchName, helper.ptrType, loc,
                         operands);
    call->setDiscardableAttr(mlir::gc::GC_ATTR_KERNEL_NAME,
                             launch.getKernelModuleName());
    helper.kernelPtrArgs[call.getOperation()] = std::move(ptrArgs);
    rewriter.eraseOp(launch);
    return success();
  }

private:
  // Returns the cached kernel pointer, creating the accessor functions and the
  // globals on the first use of this kernel.
  Value getKernel(gpu::LaunchFuncOp launch,
                  ConversionPatternRewriter &rewriter) const {
    auto mod = launch->getParentOfType<ModuleOp>();
    SmallString<128> getterName(KERNEL_GETTER);
    getterName.append(launch.getKernelModuleName().getValue());
    auto getter = mod.lookupSymbol<LLVM::LLVMFuncOp>(getterName);

    if (!getter) {
      OpBuilder::InsertionGuard guard(rewriter);
      getter = createAccessors(launch, rewriter, mod, getterName);
      if (!getter) return {};
    }

    return LLVM::CallOp::create(rewriter, launch.getLoc(), getter,
                                ValueRange{helper.runtime})
        .getResult();
  }

  // Creates the kernel globals, the gcGpuCreateKernel_<kernel>() function,
  // that creates the kernel and stores the pointer in the global variable, and
  // the gcGpuGetKernel_<kernel>() function, that returns the cached pointer or
  // calls the create function. Both are thread-safe.
  LLVM::LLVMFuncOp createAccessors(gpu::LaunchFuncOp launch,
                                   ConversionPatternRewriter &rewriter,
                                   ModuleOp mod, StringRef getterName) const {
    auto loc = launch.getLoc();
    auto binary = getBinary(launch);
    if (!binary) return {};

    auto kernelModName = launch.getKernelModuleName().getValue();
    SmallString<128> buf(KERNEL_GLOBAL);
    buf.append(kernelModName);
    buf.append("_");
    auto prefixLen = buf.size();
    auto name = [&](const char *suffix) -> SmallString<128> & {
      buf.truncate(prefixLen);
      buf.append(suffix);
      return buf;
    };
    auto funcType =
        LLVM::LLVMFunctionType::get(helper.ptrType, {helper.ptrType});

    rewriter.setInsertionPointToStart(mod.getBody());
    // The kernel pointer is cached here. The initial value must be null, the
    // getter checks it to create the kernel on the first use.
    auto global = LLVM::GlobalOp::create(
        rewriter, loc, helper.ptrType, /*isConstant=*/false,
        LLVM::Linkage::Internal, name("Ptr"),
        LLVM::ZeroAttr::get(rewriter.getContext()));
    helper.kernels.emplace_back(global);

    SmallVector<uint8_t> argTypes;
    using ArgType = ::gc::gpu::ArgType;
    auto toArgType = [&](Type orig, Type converted) -> std::optional<ArgType> {
      if (isa<LLVM::LLVMPointerType>(converted)) return ArgType::PTR;
      auto type = isa<IntegerType, FloatType>(orig) ? orig : converted;
      if (auto integer = dyn_cast<IntegerType>(type)) {
        auto u = integer.isUnsigned();
        switch (integer.getWidth()) {
        case 8:
          return u ? ArgType::UI8 : ArgType::I8;
        case 16:
          return u ? ArgType::UI16 : ArgType::I16;
        case 32:
          return u ? ArgType::UI32 : ArgType::I32;
        case 64:
          return u ? ArgType::UI64 : ArgType::I64;
        }
        return std::nullopt;
      }
      if (isa<FloatType>(type)) {
        switch (type.getIntOrFloatBitWidth()) {
        case 32:
          return ArgType::F32;
        case 64:
          return ArgType::F64;
        }
      }
      return std::nullopt;
    };
    for (auto operand : launch.getKernelOperands()) {
      auto orig = operand.getType();
      auto argType = toArgType(orig, helper.converter.convertType(orig));
      if (!argType) {
        launch.emitOpError() << "unsupported kernel argument type " << orig
                             << ": expected a pointer or a scalar";
        return {};
      }
      argTypes.emplace_back(static_cast<uint8_t>(*argType));
    }

    LLVM::GlobalOp argTypesGlobal;
    if (!argTypes.empty()) {
      auto i8Type = IntegerType::get(rewriter.getContext(), 8);
      auto arrayType = LLVM::LLVMArrayType::get(i8Type, argTypes.size());
      auto tensorType = RankedTensorType::get(
          {static_cast<int64_t>(argTypes.size())}, i8Type);
      argTypesGlobal = LLVM::GlobalOp::create(
          rewriter, loc, arrayType, /*isConstant=*/true,
          LLVM::Linkage::Internal, name("ArgTypes"),
          DenseElementsAttr::get(tensorType, ArrayRef(argTypes)));
    }

    SmallString<128> creatorName(KERNEL_CREATOR);
    creatorName.append(kernelModName);
    auto creator = LLVM::LLVMFuncOp::create(rewriter, loc, creatorName,
                                            funcType, LLVM::Linkage::Internal);
    rewriter.setInsertionPointToStart(creator.addEntryBlock(rewriter));
    {
      auto nArgs =
          helper.constant(rewriter, loc, helper.i32Type, argTypes.size());
      auto argTypesPtr =
          argTypes.empty()
              ? helper.nullPtr(rewriter, loc)
              : LLVM::AddressOfOp::create(rewriter, loc, argTypesGlobal);
      auto kernelName = launch.getKernelName().getValue();
      SmallVector<char> chars(kernelName.begin(), kernelName.end());
      chars.emplace_back('\0');
      // The kernel name and the binary are stored as global strings.
      auto namePtr = LLVM::createGlobalString(
          loc, rewriter, name("Name"), StringRef(chars.data(), chars.size()),
          LLVM::Linkage::Internal);
      auto binPtr =
          LLVM::createGlobalString(loc, rewriter, name("Bin"),
                                   binary.getValue(), LLVM::Linkage::Internal);
      auto kernel =
          funcCall(
              rewriter, RtSym::gcGpuCreateKernelName, helper.ptrType, loc,
              {creator.getArgument(0), binPtr,
               helper.constant(rewriter, loc, helper.sizeType, binary.size()),
               namePtr, nArgs, argTypesPtr})
              .getResult();

      // Cache the pointer using CAS.
      auto cas = LLVM::AtomicCmpXchgOp::create(
          rewriter, loc, LLVM::AddressOfOp::create(rewriter, loc, global),
          helper.nullPtr(rewriter, loc), kernel, LLVM::AtomicOrdering::acq_rel,
          LLVM::AtomicOrdering::monotonic);
      auto stored = LLVM::ExtractValueOp::create(
          rewriter, loc, rewriter.getI1Type(), cas, ArrayRef<int64_t>{1});

      auto body = &creator.getBody();
      auto thenBlock = rewriter.createBlock(body);
      auto elseBlock = rewriter.createBlock(body);
      rewriter.setInsertionPointToEnd(&body->front());
      LLVM::CondBrOp::create(rewriter, loc, stored, thenBlock, elseBlock);

      rewriter.setInsertionPointToStart(thenBlock);
      LLVM::ReturnOp::create(rewriter, loc, kernel);

      // The kernel has already been created by another thread, destroying this
      // one and using the cached pointer.
      rewriter.setInsertionPointToStart(elseBlock);
      destroyKernel(rewriter, loc, kernel);
      LLVM::ReturnOp::create(rewriter, loc,
                             LLVM::ExtractValueOp::create(rewriter, loc,
                                                          helper.ptrType, cas,
                                                          ArrayRef<int64_t>{0})
                                 .getResult());
    }

    rewriter.setInsertionPointAfter(creator);
    auto getter = LLVM::LLVMFuncOp::create(rewriter, loc, getterName, funcType,
                                           LLVM::Linkage::Internal);
    getter.setAlwaysInline(true);
    rewriter.setInsertionPointToStart(getter.addEntryBlock(rewriter));
    {
      // The acquire load is paired with the release in the CAS above, ensuring
      // that the kernel object, created by another thread, is visible here.
      auto kernel = LLVM::LoadOp::create(
          rewriter, loc, helper.ptrType,
          LLVM::AddressOfOp::create(rewriter, loc, global),
          helper.converter.getPointerBitwidth() / 8, /*isVolatile=*/false,
          /*isNonTemporal=*/false, /*isInvariant=*/false,
          /*isInvariantGroup=*/false, /*isNoundef=*/false,
          LLVM::AtomicOrdering::acquire);
      auto isNull = LLVM::ICmpOp::create(rewriter, loc, LLVM::ICmpPredicate::eq,
                                         kernel, helper.nullPtr(rewriter, loc));

      auto body = &getter.getBody();
      auto thenBlock = rewriter.createBlock(body);
      auto elseBlock = rewriter.createBlock(body);
      rewriter.setInsertionPointToEnd(&body->front());
      LLVM::CondBrOp::create(rewriter, loc, isNull, thenBlock, elseBlock);

      rewriter.setInsertionPointToStart(thenBlock);
      LLVM::ReturnOp::create(
          rewriter, loc,
          LLVM::CallOp::create(rewriter, loc, creator,
                               ValueRange{getter.getArgument(0)})
              .getResult());

      rewriter.setInsertionPointToStart(elseBlock);
      LLVM::ReturnOp::create(rewriter, loc, kernel.getResult());
    }

    return getter;
  }

  StringAttr getBinary(gpu::LaunchFuncOp launch) const {
    auto kernelModName = launch.getKernelModuleName();
    auto binaryOp = SymbolTable::lookupNearestSymbolFrom<gpu::BinaryOp>(
        launch, kernelModName);
    if (!binaryOp) {
      launch.emitOpError() << "binary " << kernelModName << " not found";
      return {};
    }
    auto objects = binaryOp.getObjects();
    auto object =
        objects.empty() ? nullptr : dyn_cast<gpu::ObjectAttr>(objects[0]);
    auto binary = object ? object.getObject() : nullptr;
    if (!binary) binaryOp.emitOpError() << "missing binary";
    return binary;
  }

  void destroyKernel(OpBuilder &rewriter, Location loc, Value kernel) const {
    funcCall(rewriter, RtSym::gcGpuDestroyKernelName, helper.voidType, loc,
             kernel);
  }
};

// mgpuMemAlloc(size, stream, isHostShared) -> gcGpuAlloc*(runtime, size, align)
struct ConvertAlloc final : ConvertCall {
  using ConvertCall::ConvertCall;

  LogicalResult matchAndRewrite(LLVM::CallOp call,
                                PatternRewriter &rewriter) const override {
    if (!isCall(call, "mgpuMemAlloc") || call.getNumOperands() != 3)
      return failure();

    auto loc = call.getLoc();
    auto shared = getConstantIntValue(call.getOperand(2));
    auto name = !shared || !*shared ? RtSym::gcGpuAllocDeviceName
                                    : RtSym::gcGpuAllocSharedName;
    auto size =
        helper.toInt(rewriter, loc, call.getOperand(0), helper.sizeType);
    if (!size) return call.emitOpError() << "the size must be an integer";
    // The alignment is selected by the runtime.
    auto align = helper.constant(rewriter, loc, helper.sizeType, 0);
    rewriter.replaceOp(call, funcCall(rewriter, name, helper.ptrType, loc,
                                      {helper.runtime, size, align})
                                 .getResult());
    return success();
  }
};

// mgpuMemFree(ptr, stream) -> gcGpuFree(runtime, ptr)
struct ConvertFree final : ConvertCall {
  using ConvertCall::ConvertCall;

  LogicalResult matchAndRewrite(LLVM::CallOp call,
                                PatternRewriter &rewriter) const override {
    if (!isCall(call, "mgpuMemFree") || call.getNumOperands() != 2)
      return failure();
    auto ptr = call.getOperand(0);
    if (!isPtr(ptr)) return call.emitOpError() << "expected a pointer argument";
    funcCall(rewriter, RtSym::gcGpuFreeName, helper.voidType, call.getLoc(),
             {helper.runtime, ptr});
    rewriter.eraseOp(call);
    return success();
  }
};

// mgpuMemcpy(dst, src, size, stream) ->
// gcGpuMemcpy(runtime, src, dst, size, deps, nDeps)
struct ConvertMemcpy final : ConvertCall {
  using ConvertCall::ConvertCall;

  LogicalResult matchAndRewrite(LLVM::CallOp call,
                                PatternRewriter &rewriter) const override {
    if (!isCall(call, "mgpuMemcpy") || call.getNumOperands() != 4)
      return failure();

    auto loc = call.getLoc();
    auto dst = call.getOperand(0);
    auto src = call.getOperand(1);
    if (!isPtr(dst) || !isPtr(src))
      return call.emitOpError() << "expected pointer arguments";
    auto size =
        helper.toInt(rewriter, loc, call.getOperand(2), helper.sizeType);
    if (!size) return call.emitOpError() << "the size must be an integer";
    // The event dependencies are merged into the external ones, if any, after
    // all the operations are converted.
    funcCall(rewriter, RtSym::gcGpuMemcpyName, helper.ptrType, loc,
             {helper.runtime, src, dst, size, helper.deps, helper.nDeps});
    rewriter.eraseOp(call);
    return success();
  }
};

// Tracks the events, produced by the gcGpuLaunch and gcGpuMemcpy calls, and
// passes them as dependencies to the operations, that must not start before
// the event is complete. The events, that are passed to gcGpuWait(), are
// released - the ownership is transferred to the runtime.
struct Dependencies final {
  const Helper &helper;
  ModuleOp mod;

  struct Event {
    Operation *op;
    Value event;
    // The kernel module name or empty for gcGpuMemcpy.
    StringRef kernel;
    // The kernel pointer arguments or {src, dst} for gcGpuMemcpy.
    SmallVector<Value> ptrs;
    bool released = false;

    bool isMemcpy() const { return kernel.empty(); }
    Value dst() const { return ptrs[1]; }
    bool uses(Value ptr) const { return llvm::is_contained(ptrs, ptr); }
    bool usesAny(ValueRange values) const {
      return llvm::any_of(values, [&](Value v) { return uses(v); });
    }
  };

  SmallVector<Event> events;

  Dependencies(const Helper &helper, ModuleOp mod) : helper(helper), mod(mod) {}

  // Resolves the dependencies of all the runtime calls in the block and
  // returns the events we still own. Only the events, created before the
  // current operation, are considered.
  SmallVector<Value> resolve(Block &block) {
    collect(block);
    SmallVector<CallDeps> calls;
    size_t created = 0;
    for (auto &op : block) {
      if (created < events.size() && events[created].op == &op) {
        auto &event = events[created];
        auto deps = event.isMemcpy() ? memcpyDeps(event, created)
                                     : launchDeps(event, created);
        if (!deps.empty())
          calls.emplace_back(cast<LLVM::CallOp>(&op), std::move(deps));
        created++;
      } else if (isAlloc(&op)) {
        // Do not allocate the memory for a kernel, that is going to wait for
        // other events - this may result in OOM.
        for (auto &event : llvm::drop_begin(events, created)) {
          if (!event.isMemcpy() && event.uses(op.getResult(0))) {
            waitFor(&op, launchDeps(event, created));
            break;
          }
        }
      } else if (isCall(&op, RtSym::gcGpuFreeName)) {
        SmallVector<Value> deps;
        for (auto &event : ArrayRef(events).take_front(created))
          if (!event.released && event.uses(op.getOperand(1)))
            deps.emplace_back(event.event);
        waitFor(&op, deps);
      }
    }

    helper.setDeps(block, calls);

    SmallVector<Value> owned;
    for (auto &event : events)
      if (!event.released) owned.emplace_back(event.event);
    return owned;
  }

private:
  static bool isAlloc(Operation *op) {
    return isCall(op, RtSym::gcGpuAllocDeviceName) ||
           isCall(op, RtSym::gcGpuAllocSharedName) ||
           isCall(op, RtSym::gcGpuAllocHostName);
  }

  void collect(Block &block) {
    for (auto &op : block) {
      if (isCall(&op, RtSym::gcGpuLaunchName)) {
        auto name = cast<StringAttr>(
                        op.getDiscardableAttr(mlir::gc::GC_ATTR_KERNEL_NAME))
                        .getValue();
        events.emplace_back(Event{&op, op.getResult(0), name,
                                  helper.kernelPtrArgs.lookup(&op)});
      } else if (isCall(&op, RtSym::gcGpuMemcpyName)) {
        events.emplace_back(Event{
            &op, op.getResult(0), {}, {op.getOperand(1), op.getOperand(2)}});
      }
    }
  }

  // A kernel depends on the kernels, listed in the module attributes, and on
  // the copies, writing to its arguments.
  SmallVector<Value> launchDeps(const Event &launch, size_t created) const {
    auto depends = mlir::gc::KernelAttrs(mod, launch.kernel).getDepends();
    SmallVector<Value> deps;
    for (auto &event : ArrayRef(events).take_front(created)) {
      if (event.released) continue;
      if (event.isMemcpy()
              ? launch.uses(event.dst())
              : depends && llvm::is_contained(*depends, event.kernel))
        deps.emplace_back(event.event);
    }
    return deps;
  }

  // A copy depends on the kernels, accessing the source or the destination.
  SmallVector<Value> memcpyDeps(const Event &memcpy, size_t created) const {
    SmallVector<Value> deps;
    for (auto &event : ArrayRef(events).take_front(created))
      if (!event.released && !event.isMemcpy() && event.usesAny(memcpy.ptrs))
        deps.emplace_back(event.event);
    return deps;
  }

  void waitFor(Operation *op, ArrayRef<Value> deps) {
    if (deps.empty()) return;
    OpBuilder rewriter(op);
    helper.wait(rewriter, op->getLoc(), deps);
    for (auto &event : events)
      if (llvm::is_contained(deps, event.event)) event.released = true;
  }
};

struct GpuToGcGpu final : mlir::gc::impl::GpuToGcGpuBase<GpuToGcGpu> {

  explicit GpuToGcGpu() : GpuToGcGpu(mlir::gc::GpuToGcGpuOptions{}) {}
  explicit GpuToGcGpu(const mlir::gc::GpuToGcGpuOptions &opts)
      : GpuToGcGpuBase(opts) {}

  void runOnOperation() override {
    auto mod = cast<ModuleOp>(getOperation());
    auto ctx = &getContext();
    auto entry = getEntryFunc(mod);
    if (!entry) {
      signalPassFailure();
      return;
    }

    LLVMTypeConverter converter(ctx);
    Type ptrType = LLVM::LLVMPointerType::get(ctx);
    Type i32Type = IntegerType::get(ctx, 32);
    // The runtime pointer and the external dependencies are appended to the
    // arguments: (..., runtime, deps, nDeps).
    for (auto type : {ptrType, ptrType, i32Type}) {
      if (failed(entry.insertArgument(entry.getNumArguments(), type, {},
                                      entry.getLoc()))) {
        entry.emitOpError() << "failed to append the runtime arguments";
        signalPassFailure();
        return;
      }
    }
    auto args = entry.getNumArguments();
    Helper helper(ctx, converter, &entry.getBody().front(),
                  entry.getArgument(args - 3), entry.getArgument(args - 2),
                  entry.getArgument(args - 1));

    {
      RewritePatternSet patterns(ctx);
      patterns.add<ConvertAlloc, ConvertFree, ConvertMemcpy>(ctx, helper);
      if (failed(applyPatternsGreedily(mod, std::move(patterns)))) {
        signalPassFailure();
        return;
      }
    }
    {
      const LLVMConversionTarget target(*ctx);
      RewritePatternSet patterns(ctx);
      populateGpuToLLVMConversionPatterns(converter, patterns);
      patterns.add<ConvertLaunch>(helper);
      if (failed(applyPartialConversion(mod, target, std::move(patterns)))) {
        signalPassFailure();
        return;
      }
    }

    if (failed(eraseUnused(mod))) {
      signalPassFailure();
      return;
    }

    Dependencies dependencies(helper, mod);
    auto events = dependencies.resolve(entry.getBody().front());
    if (failed(returnEvents(entry, helper, events))) {
      signalPassFailure();
      return;
    }
    helper.finalize();

    // Only the Program::Symbols functions are public.
    createEventNum(mod, helper, wait ? 0 : events.size());
    createDestructor(mod, helper);
    if (failed(rename(mod, entry))) signalPassFailure();
  }

private:
  // The module is expected to contain a single externally visible function
  // definition, containing all the GPU operations. Everything, created by this
  // pass, except the Program::Symbols functions, has internal linkage.
  LLVM::LLVMFuncOp getEntryFunc(ModuleOp mod) {
    LLVM::LLVMFuncOp entry;
    for (auto func : mod.getOps<LLVM::LLVMFuncOp>()) {
      if (func.isExternal() || func.getLinkage() != LLVM::Linkage::External)
        continue;
      if (entry) {
        func.emitOpError() << "the module must contain a single public "
                              "function, but "
                           << entry.getSymName() << " is also public";
        return {};
      }
      entry = func;
    }

    if (!entry) {
      mod.emitOpError() << "no public function found";
      return {};
    }

    for (auto name :
         {ProgSym::mainName, ProgSym::eventNumName, ProgSym::destructorName}) {
      if (auto *op = mod.lookupSymbol(name); op && op != entry.getOperation()) {
        op->emitOpError() << "the symbol name " << name
                          << " is reserved for the generated function";
        return {};
      }
    }

    auto walk = mod.walk([&](Operation *op) {
      if ((isa<gpu::LaunchFuncOp>(op) || isMgpuCall(op)) &&
          op->getParentOfType<LLVM::LLVMFuncOp>() != entry) {
        op->emitOpError() << "the GPU operations are only supported in the "
                             "entry function "
                          << entry.getSymName();
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    return walk.wasInterrupted() ? nullptr : entry;
  }

  // Renames the entry function to Program::Symbols::main.
  LogicalResult rename(ModuleOp mod, LLVM::LLVMFuncOp entry) {
    auto name = StringAttr::get(mod.getContext(), ProgSym::mainName);
    if (entry.getSymNameAttr() == name) return success();
    if (failed(SymbolTable::replaceAllSymbolUses(entry, name, mod))) {
      return entry.emitOpError() << "failed to rename to " << ProgSym::mainName;
    }
    entry.setSymName(name);
    return success();
  }

  // Deletes the mgpu* calls, that have not been converted, and their
  // declarations, and the GPU binaries, that are embedded in the module.
  LogicalResult eraseUnused(ModuleOp mod) {
    SmallVector<Operation *> calls;
    mod.walk([&](LLVM::CallOp call) {
      if (isMgpuCall(call)) calls.emplace_back(call);
    });

    // A call can only be erased after its users, that are erased as well.
    for (bool erased = true; erased;) {
      erased = false;
      for (auto *&call : calls) {
        if (call && call->use_empty()) {
          call->erase();
          call = nullptr;
          erased = true;
        }
      }
    }
    for (auto *call : calls)
      if (call) return call->emitOpError() << "unsupported GPU runtime call";

    for (auto func : llvm::make_early_inc_range(mod.getOps<LLVM::LLVMFuncOp>()))
      if (func.isExternal() && isMgpu(func.getSymName()) &&
          SymbolTable::symbolKnownUseEmpty(func, mod))
        func.erase();

    // The binaries are copied to the module globals by ConvertLaunch.
    for (auto binary : llvm::make_early_inc_range(mod.getOps<gpu::BinaryOp>()))
      if (binary.use_empty()) binary.erase();
    return success();
  }

  // If the wait option is set, waits for all the events we own. Otherwise,
  // appends an argument, pointing to the events array, and stores the events
  // there, transferring the ownership to the caller.
  LogicalResult returnEvents(LLVM::LLVMFuncOp entry, const Helper &helper,
                             ArrayRef<Value> events) {
    SmallVector<LLVM::ReturnOp> returns;
    entry.walk([&](LLVM::ReturnOp ret) { returns.emplace_back(ret); });

    if (!events.empty()) {
      DominanceInfo dom(entry);
      for (auto ret : returns) {
        for (auto event : events) {
          if (dom.dominates(event.getDefiningOp(), ret.getOperation()))
            continue;
          return ret.emitOpError()
                 << "the GPU operations must dominate all the return "
                    "operations of the entry function";
        }
      }
    }

    if (wait) {
      if (events.empty()) return success();
      for (auto ret : returns) {
        OpBuilder rewriter(ret);
        helper.wait(rewriter, ret.getLoc(), events);
      }
      return success();
    }

    if (events.empty()) return success();
    auto index = entry.getNumArguments();
    if (failed(entry.insertArgument(index, helper.ptrType, {}, entry.getLoc())))
      return entry.emitOpError() << "failed to append the events argument";
    auto arr = entry.getArgument(index);
    for (auto ret : returns) {
      OpBuilder rewriter(ret);
      auto loc = ret.getLoc();
      for (auto [i, event] : llvm::enumerate(events))
        LLVM::StoreOp::create(rewriter, loc, event,
                              helper.slot(rewriter, loc, arr, i));
    }
    return success();
  }

  // Adds the eventNum() function, returning the number of the events, the main
  // function stores to the array, passed as its last argument.
  void createEventNum(ModuleOp mod, const Helper &helper, unsigned eventNum) {
    OpBuilder rewriter(mod->getContext());
    auto loc = mod.getLoc();
    createPublicFunc(rewriter, mod, ProgSym::eventNumName, helper.i32Type);
    LLVM::ReturnOp::create(
        rewriter, loc,
        helper.constant(rewriter, loc, helper.i32Type, eventNum));
  }

  // Adds the destructor() function, that destroys all the kernels.
  void createDestructor(ModuleOp mod, const Helper &helper) {
    OpBuilder rewriter(mod->getContext());
    auto loc = mod.getLoc();
    createPublicFunc(rewriter, mod, ProgSym::destructorName, helper.voidType);
    if (!helper.kernels.empty()) {
      // Paired with the release CAS in the kernel creators.
      LLVM::FenceOp::create(rewriter, loc, LLVM::AtomicOrdering::acquire);
    }

    for (auto global : helper.kernels) {
      auto kernel = LLVM::LoadOp::create(
          rewriter, loc, helper.ptrType,
          LLVM::AddressOfOp::create(rewriter, loc, global));
      funcCall(rewriter, RtSym::gcGpuDestroyKernelName, helper.voidType, loc,
               kernel.getResult());
    }

    LLVM::ReturnOp::create(rewriter, loc, ValueRange{});
  }
};
} // namespace
