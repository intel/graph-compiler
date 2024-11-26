//===- GpuUtilsTest.cpp - Tests for GpuUtils-------------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <iostream>

#include "gc/Transforms/GPU/GpuUtils.h"

#include "gtest/gtest.h"

TEST(testAdjustTiles, GputUtilsTest) {
  bool print = false;
  auto testAdjust = [print](int64_t totalSize, SmallVector<int64_t> &tiles,
                            const SmallVector<int64_t> &expected) {
    if (print) {
      std::cout << totalSize << ": [";
      for (unsigned i = 0; i < tiles.size(); i++) {
        std::cout << tiles[i] << (i + 1 < tiles.size() ? ", " : "");
      }
      std::cout << "] -> [";
    }

    gc::adjustTiles(totalSize, tiles);

    if (print) {
      for (unsigned i = 0; i < tiles.size(); i++) {
        std::cout << tiles[i] << (i + 1 < tiles.size() ? ", " : "");
      }
      std::cout << "]" << std::endl;
    }

    EXPECT_EQ(tiles, expected);
  };
  auto test = [testAdjust](int64_t totalSize, SmallVector<int64_t> tiles,
                           SmallVector<int64_t> expected) {
    if (tiles.size() != 2 || tiles[0] == tiles[1]) {
      testAdjust(totalSize, tiles, expected);
      return;
    }
    SmallVector<int64_t> reversed(tiles.rbegin(), tiles.rend());
    testAdjust(totalSize, tiles, expected);
    std::reverse(expected.begin(), expected.end());
    testAdjust(totalSize, reversed, expected);
  };

  test(8, {1, 1}, {1, 1});
  test(8, {1, 2}, {1, 2});
  test(8, {2, 2}, {2, 2});
  test(8, {1, 4}, {1, 4});
  test(8, {1, 8}, {1, 8});
  test(8, {2, 8}, {2, 4});
  test(8, {1, 32}, {1, 8});
  test(8, {2, 32}, {1, 8});
  test(8, {4, 32}, {1, 8});
  test(8, {8, 32}, {2, 4});
  test(8, {16, 32}, {2, 4});
  test(8, {32, 32}, {2, 4});
  test(8, {64, 32}, {4, 2});
  test(8, {128, 32}, {4, 2});

  test(8192, {1024, 1024}, {64, 128});
  test(8192, {32, 32}, {32, 32});
  test(8192, {16, 64}, {16, 64});
  test(8192, {8, 128}, {8, 128});
  test(8192, {4, 256}, {4, 256});
  test(8192, {2, 512}, {2, 512});
  test(8192, {1, 1024}, {1, 1024});
  test(8192, {512, 2}, {512, 2});
  test(8192, {256, 4}, {256, 4});
  test(8192, {128, 8}, {128, 8});
  test(8192, {64, 16}, {64, 16});
  test(8192, {32, 32}, {32, 32});

  test(16384, {1, 1, 1}, {1, 1, 1});
  test(16384, {1, 2, 4}, {1, 2, 4});
  test(16384, {2, 4, 8}, {2, 4, 8});
  test(16384, {4, 8, 16}, {4, 8, 16});
  test(16384, {8, 16, 32}, {8, 16, 32});
  test(16384, {16, 32, 64}, {16, 32, 32});
  test(16384, {32, 64, 128}, {8, 32, 64});
  test(16384, {64, 128, 256}, {8, 32, 64});
  test(16384, {128, 256, 512}, {4, 64, 64});

  test(16384, {7, 17, 111}, {7, 17, 111});
  test(16384, {7, 117, 111}, {7, 39, 37});
  test(16384, {6, 256, 512}, {1, 128, 128});
  test(16384, {60, 128, 512}, {4, 32, 128});
  test(16384, {119, 256, 512}, {7, 32, 64});
  test(16384, {109, 256, 512}, {109, 8, 16});
}
