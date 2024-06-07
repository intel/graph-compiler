# Easy builder for building IR in C++

Sometimes in the transformations of MLIR, developers may need to use C++ to
build complex IR. For example, developers may expand math operators `exp` to
polynormial expressions when mathematical approximation is allowed. This may
result in tens of calls to `builder.create<T>(loc, ...)` in the expansion pass
implementation. Another example is that when a pass expands and tiles a `matmul`
operation, the developer may need to write lines of code to create `scf.for` and
`scf.if`, and manipulate the insertion point for the `OpBuilder` to create IR
with complex control flow to schedule the tile based on thread-id and cache
size. One can imagine that the C++ code to accomplish the above task will be
verbose and hard to read. The easy-builder utilities are introduced to
make it easier to develop C++ code building complex IR. Easy-builder is not
designed to replace the `OpBuilder`. Instead, it is built upon that, and serves
as supplementary IR builder for complex cases, like heavily using `arith` and
`scf` operations.

An example code to build IR `(x+y-10)/(x-y+1)`, where `x` and `y` are unsigned
16-bit integers:

```C++
OpBuilder builder = ...;
Value x = ...;
Value y = ...;
Location loc = ...;

// start of easy-build
EasyBuilder b{builder, loc};
EBUnsigned wx = b.wrap<EBUnsigned>(x);
EBUnsigned wy = b.wrap<EBUnsigned>(y);
EBUnsigned result = (wx + wy - uint16_t(10)) / (wx - wy + uint16_t(1));
```

The `result` above can be implicitly converted to `Value` type.

In contrast, the code using `OpBuilder` to do the same task will have more lines
of code and less readablity:

```C++
OpBuilder builder = ...;
Value x = ...;
Value y = ...;
Location loc = ...;

auto x_p_y = builder.create<arith::AddIOp>(loc, x, y);
auto v10 = builder.create<arith::ConstantIntOp>(loc, 10, /*width*/16);
auto x_p_y_m10 = builder.create<arith::SubIOp>(loc, x_p_y, v10);
auto x_m_y = builder.create<arith::SubIOp>(loc, x, y);
auto v1 = builder.create<arith::ConstantIntOp>(loc, 1, /*width*/16);
auto x_m_y_p1 = builder.create<arith::AddIOp>(loc, x_m_y, v1);
Value result = builder.create<arith::DivUIOp>(loc, x_p_y_m10, x_m_y_p1);
```

[TOC]

## Overall Design

There are some observations on the C++ code for IR creation in MLIR transforms:

1. Consecutive calls to `builder.create<T>(loc, ...)` often use the same
   `OpBuilder` and `Location`. Many mutations in the MLIR passes are to expand
   an operation into a sequence of operations. So expanded operations will share
   the same builder and the same location of the original operation.
2. There is currently no C++ operator overloading on MLIR's `Value` class.
   Developers have to break a long arithmatical expression into calls to the
   `create<..>` function.
3. It would be helpful if MLIR provides helper functions to convert C++ types
   (like `uint16_t` and `OpFoldResult`) into `Value`.
4. C++ RAII and use of macros could be helpful for building
   structured-control-flow IR (not just `scf` operations).

Easy-build is designed to improve the `OpBuilder` in the above aspects. It is
also extendable for different dialects and operations - developers can extend
easy-builder for a new dialect with reasonable efforts.

Easy-build provides `EBValue` class ("EB" here stands for "easy-build") as a
wrapper for MLIR's `Value` objects. `EBValue` also stores the reference to the
`OpBuilder` and `Location`. An `EBValue` is self-contained for creating new
operations based on it. This enables C++ operator overloading on `EBValue`. Note
that it is hard to implement operator overloading on MLIR's `Value` to build new
operation, because it lacks information of the `OpBuilder` and `Location` of the
new operation to create.

## Design details

Most of easy-build's APIs are defined in `mlir::easybuild` namespace. This
section will introduc the key data structures of easy-build.

### EasyBuildState

This class holds the "states" of an easy-builder. It contains the a `Location`
object, a reference to the `OpBuilder` and other configurations of an an
easy-builder. A shared-ptr to a `EasyBuildState` is attached to every non-empty
`EBValue`. Further creation of new operations based on a `EBValue` should use
the `OpBuilder` and `Location` in the referenced `EasyBuildState`. The newly
created `EBValue` should hold a shared-ptr to the same `EasyBuildState` of its
operands.

```c++
struct EasyBuildState {
  OpBuilder &builder;
  Location loc;
  ...
};
```

### EBValue

This class is essentially a `mlir::Value` with shared-ptr to `EasyBuildState`.
`EBValue` is a general base class for any values and itself does not enable C++
operator overloading for IR creation. However, developers can inherit this class
to restrict the `Value` to be held in a `EBValue` and enable some specific
IR-building utilities. In the example at the beginning of this document, a
subclass `EBUnsigned` is used to hold `Value` of "unsigned integer". The line
`EBUnsigned wx = b.wrap<EBUnsigned>(x);` converts `x` of `Value` to
`EBUnsigned`. If `x`'s type is not compatible to `EBUnsigned`, a runtime
assertion failure may occur. Easy-build for `arith` dialects enables operator
overloads for `EBUnsigned` like:

```c++
EBUnsigned operator+(EBUnsigned a, EBUnsigned b) {
    return EBUnsigned {a.builder,
            a.builder->builder.template create<arith::AddIOp>(a.builder->loc,
            a.v, b.v)};
}
```

The created `EBUnsigned` should share the same `EasyBuildState` pointer of the
operands.

An `EBValue` object can be implictly be converted to `Value`:

```c++
EBValue wrapped = ...;
Value v = wrapped;
```

Please refer to sections [Easy-build for arith dialect](#Easy-build-for-arith-dialect)
for the subclasses of `EBValue` for arith operations. See also
[Extending easy-build for dialects](#Extending-easy-build-for-dialects) for
extending `EBValue` for a new dialect.

### EasyBuilder

The `EasyBuilder` is a utility class for

1. creating an initial `EasyBuildState` object
2. wrapping `Value`, C++ numerical values (e.g. `uint32_t`, `float`) or
   `OpFoldResult` into `EBValue` or its subclasses, and setting the shared-ptr
   `EasyBuildState` of the created values.
3. setting `Location` for the next created operation

The use of easy-build usually starts from creating an EasyBuilder. The
constructor of it will internally create an `EasyBuildState` object with the
given `OpBuilder` and `Location`.

#### Wrapping various values to EBValue

To convert various types of C++ values to `EBValue` or its subclasses,
`EasyBuilder` provides a template function `EasyBuilder::wrap<T>(V)` to convert
C++ type `V` into `T`, which is a subclass of `EBValue`. The result `EBValue` or
its subclasses's should hold a shared-ptr pointing to the `EasyBuildState` of
this `EasyBuilder`. `EasyBuilder::wrap` may introduce runtime type checking for
the input value (implementation provided by the class `T`). When the convertion
fails, an assertion failure may happen at the runtime. An example of such case
is when we try to wrap a `memref` typed `Value` to `EBUnsigned`, the convertion
should fail, because `memref` are not arithmetic values.

`EasyBuilder` provides a similar function `EasyBuilder::wrapOrFail<T>(V)` which
returns `FailureOr<T>`. It has similar functionality of `wrap()`, except that it
returns `failure()` when the convertion fails, instead of triggering an runtime
abortion.

```C++
#include "mlir/Dialect/Arith/Utils/EasyBuild.h"

EasyBuilder b {...};
Value v = builder.create<arith::ConstantFloatOp>(...);
EBFloatPoint u1 = b.wrap<EBFloatPoint>(v); // OK
FailureOr<EBUnsigned> u1 = b.wrapOrFail<EBUnsigned>(v);
assert(failed(u1)); // convertion should fail
```

`EasyBuilder` overrides `operator()` to provide convenience converter for
general `Value` to `EBValue` and arithmetic C++ values to the corresponding
subclass of `EBValue`:

```C++
#include "mlir/Dialect/Arith/Utils/EasyBuild.h"

EasyBuilder b {...};
Value v = ...;
EBValue v1 = b(v); // wrap Value to base class EBValue
EBUnsigned u1 = b(uint32_t(2)); // creating arith.constant of i32
EBFloatPoint u1 = b(2.0f); // creating arith.constant of f32
```

#### Setting source location

Users can set the `Location` to be used in the `OpBuilder::create` after a call
to `EasyBuilder::setLoc()`. New operations related to a `EasyBuildState` will be
created with the new `Location` set by `EasyBuilder::setLoc()`. The previously
created operations' location before calling `setLoc()` will not be changed.

#### Creating operations using EBValues as inputs

Developers can call the template member function `F<TOp, TValue>(...)` of
`EasyBuilder` to create a new operation of type `TOp` and wrap the result to
type `TValue`, which is `EBValue` or its subclasses. This method is used to
generate general operations for `EBValue`s. The operation will be created with
the current `OpBuilder` and `Location` of the `EasyBuilder`. For example, to
create an `mydialect::MyOp` operation with given `EBValue` as operands and get
the single result value of the operation as `EBUnsigned`:

```c++
EasyBuilder b {...};
EBValue v1 = ...;
EBUnsigned v2 = b.F<mydialect::MyOp, EBUnsigned>(v1);
```

## Typical workflow for using easy-build

To use easy-build, a developer may first include the easy-build header and
optionally include the header for the subclass of `EBValue` for a dialect.

```C++
#include "mlir/IR/EasyBuild.h"
#include "mlir/Dialect/Arith/Utils/EasyBuild.h"
```

Then in the code building the IR, create a `EasyBuilder` with an existing
`OpBuilder` and `Location`:

```c++
using namespace easybuild;
Operation* originOp = ...;
OpBuilder builder {originOp};
Location loc = originOp->getLoc();
EasyBuilder b {builder, loc};
```

Wrap `Value` or other C++ values to `EBValue` or its subclasses:

```c++
auto input1 = b.wrap<EBUnsigned>(originOp->getOperand(0));
auto input2 = b.wrap<EBUnsigned>(originOp->getOperand(1));
```

Generate operations via the wrapped values. The insert point and `Location` is
defined by the `OpBuilder` inside of `EasyBuildState`. The results can be used
as `Value`:

```c++
Value result = input1 + input2;
```

## Easy-build for arith dialect

Subclasses of `EBValue` have been defined for `arith` operations, including
`EBUnsigned`, `EBSigned` and `EBFloatPoint`. These subclasses can accept
`EasyBuilder::wrap()` of input values of types of corresponding scalar types, or
their vector or tensor type. `EBUnsigned` accepts scalar, vector or tensor of
unsigned or signless integer-or-index-typed `Value`. `EBUnsigned` accepts
scalar, vector or tensor of signed or signless integer-typed `Value`.
`EBFloatPoint` accepts scalar, vector or tensor of float-point-typed `Value`. 

Developers can also wrap C++ arithmetic types (e.g. `uint32_t`, `float`) to the
corresponding `EBUnsigned`, `EBSigned` or `EBFloatPoint` type, via
`EasyBuilder::operator()`. A call to such function will generate an
`arith.constant` operation at the current insertion point.

Similarly, `EBUnsigned`, `EBSigned` and `EBFloatPoint` enables
`EasyBuilder::wrap()` to convert from `OpFoldResult`.

```c++
OpFoldResult f = ...;
auto input1 = b.wrap<EBUnsigned>(f);
```

If `OpFoldResult` contains a `Value`, `wrap<EBUnsigned>` will try to convert the
extracted value to `EBUnsigned`. If it contains a constant as `Attr`,
`wrap<EBUnsigned>` will create an `arith.constant` based on the type of the
`Attr`.

Some of the C++ operators are enabled for `EBUnsigned`, `EBSigned` and
`EBFloatPoint` classes, include arithmetic `+ - * / % `, logical `& | ^`,
integer-shifting `>> <<` and comparison `> >= < <= == !=`. Using these C++
operators will create the corresponding `arith` operations at the current
`OpBuilder` in the `EasyBuildState` of the `EBValue`. Signed and Unsigned
integers are distinguished, so that they will emit different operations for the
arith operations that is sensitive to the signess, like `divsi` or `divui`.

## Easy-build for general structured-control-flow

TBD

## Extending easy-build for dialects

To extend easy-build for new dialects or operations, developers usually need to
create a new class to inherit the `EBValue` class and define utility helper
functions for that new class. The definition of a subclass of `EBValue` should
be operation-centric. That is, the developer of a subclass of `EBValue` should
consider which operations are designed to be applied on it, instead of
considering the data type of the `Value` first. For example, when adding support
for `arith` operations, we find that the operations of it can fall into three
categories: unsigned int, signed int and float point. Thus `EBUnsigned`,
`EBSigned` and `EBFloatPoint` classes are introduced.

The developer needs to implement the `wrapOrFail` static member function in the
subclass, to enable converting values to it via `EasyBuilder::wrap<>()`. An
example for implementing a hypothesis subclass `EBMyFloatPoint` can be:

```c++
struct EBMyFloatPoint : EBValue {
  static FailureOr<EBMyFloatPoint> wrapOrFail(const impl::StatePtr &state,
                                            Value v) {
    ...
  }
  static FailureOr<EBMyFloatPoint> wrapOrFail(const impl::StatePtr &state,
                                            const OpFoldResult &v) {
    ...
  }

  using EBValue::EBValue;
};
```

Developers can implement the utility functions to help to build the IR:

```c++
inline EBMyFloatPoint sin(EBMyFloatPoint input) {
   std::shared_ptr<EasyBuildState> state = input.builder;
   OpBuilder& builder = state->builder;
   return EBMyFloatPoint{ state,
      builder.create<MyDialect::SinOp>(state->loc, input) };
}

inline EBMyFloatPoint operator+(EBMyFloatPoint a, EBMyFloatPoint b) {
   std::shared_ptr<EasyBuildState> state = a.builder;
   OpBuilder& builder = state->builder;
   return EBMyFloatPoint{ state,
      builder.create<MyDialect::AddOp>(state->loc, a, b) };
}
```

