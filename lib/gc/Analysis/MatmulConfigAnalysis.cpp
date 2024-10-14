//===-- MatmulConfigAnalysis.cpp - Analysis for matmul config ---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc/Analysis/MatmulConfigAnalysis.h"
#include "gc/Analysis/TargetDescriptionAnalysis.h"
#include <limits>
#include <llvm/Support/Debug.h>

namespace mlir {
namespace gc {

#define DEBUG_TYPE "matmul-config-analysis"
#define LLVM_DEBUG(x) (x)

llvm::raw_ostream &operator<<(llvm::raw_ostream &ss,
                              const MatmulConfig &config) {

  ss << "{MThreads = " << config.MThreads
     << ": i32, NThreads = " << config.NThreads
     << ": i32, KThreads = " << config.KThreads
     << ": i32, MBlock = " << config.MBlock
     << ": i32, NBlock = " << config.NBlock
     << ": i32, KBlock = " << config.KBlock
     << ": i32, innermostMBlock = " << config.innerMostMBlock
     << ": i32, innermostNBlock = " << config.innerMostNBlock
     << ": i32, innermostKBlock = " << config.innerMostKBlock << ": i32}\n";
  return ss;
}

template <typename T>
static llvm::raw_ostream &operator<<(llvm::raw_ostream &ss,
                                     std::vector<T> array) {
  ss << "[";
  llvm::interleaveComma(array, ss);
  ss << "]";
  return ss;
}

bool validateConfig(const MatmulConfig &cfg, ArrayRef<uint32_t> shape,
                    bool allowIndivisibleInnerblock, bool isVNNIMM2D) {
  if (cfg.MThreads <= 0 || cfg.NThreads <= 0 || cfg.KThreads <= 0 ||
      cfg.MBlock <= 0 || cfg.NBlock <= 0 || cfg.KBlock <= 0 ||
      cfg.innerMostMBlock <= 0 || cfg.innerMostNBlock <= 0 ||
      cfg.innerMostKBlock <= 0)
    return false;
  if (cfg.MBlock % cfg.innerMostMBlock != 0 ||
      (shape[0] % cfg.innerMostMBlock != 0 && !allowIndivisibleInnerblock))
    return false;
  if (cfg.NBlock % cfg.innerMostNBlock != 0 ||
      ((shape[1] % cfg.innerMostNBlock != 0) && !allowIndivisibleInnerblock) ||
      (shape[1] % cfg.NThreads != 0 && isVNNIMM2D &&
       cfg.NBlock != cfg.innerMostNBlock))
    return false;
  // Require K % KBlock == 0 as brgemm dynamic bs is not supported now
  if (cfg.KBlock % cfg.innerMostKBlock != 0 ||
      ((shape[2] / cfg.KThreads % cfg.KBlock != 0 ||
        shape[2] / cfg.KThreads % cfg.innerMostKBlock != 0) &&
       !allowIndivisibleInnerblock))
    return false;
  // KThreads will not shrink automatically
  if (llvm::divideCeil(shape[2], cfg.KBlock) < cfg.KThreads)
    return false;
  return true;
}

// generate the candidate for the block size(factor of `num`, pow of 2 which is
// less than `num`)
std::vector<uint32_t>
getCandidate(uint32_t num, uint32_t floor,
             uint32_t ceil = std::numeric_limits<uint32_t>::max()) {
  int defaultBlock = 32;
  // factor
  std::vector<uint32_t> candidates;
  uint32_t upperbound =
      std::min(llvm::divideCeil(num, defaultBlock) * defaultBlock, ceil);
  for (uint32_t i = floor; i <= upperbound; i++)
    if (num % i == 0)
      candidates.push_back(i);

  // the pow of 2
  uint32_t candidate = 1U;
  while (candidate < floor)
    candidate *= 2;
  while (candidate <= upperbound) {
    candidates.push_back(candidate);
    candidate *= 2;
  }
  // In case that no config is valid
  candidates.push_back(num);
  // remove duplicate candidates
  std::sort(candidates.begin(), candidates.end());
  candidates.erase(std::unique(candidates.begin(), candidates.end()),
                   candidates.end());
  return candidates;
}

// check if the threads are valid
bool validateThreads(ArrayRef<uint32_t> threads,
                     CPUTargetDescriptionAnalysis &sysDesc) {
  uint32_t numThreads = sysDesc.getNumThreads();
  uint32_t actualThreads = 1U;
  for (uint32_t t : threads)
    actualThreads *= t;
  return actualThreads == numThreads;
}

struct CostModelOption {
  ArrayRef<uint32_t> shape;
  MatmulConfig cfg;
  CPUTargetDescriptionAnalysis sysDesc;
  bool verbose = false;
};

using CostModelFn =
    std::function<double(linalg::LinalgOp &linalgOp, CostModelOption option)>;

// calculate the cost of the hardware efficiency(whether the vector register is
// fully utilized)
double vectorRegEfficiencyCost(linalg::LinalgOp &linalgOp,
                               CostModelOption option) {
  MatmulConfig config = option.cfg;
  size_t dtypeSize = DataLayout().getTypeSizeInBits(
      ShapeAdaptor(linalgOp.getDpsInputs()[1].getType()).getElementType());
  size_t maxVectorWidth = option.sysDesc.getMaxVectorWidth() / dtypeSize;
  // TODO: take matrix register like amx into account
  double cost = (maxVectorWidth - config.innerMostMBlock % maxVectorWidth) %
                    maxVectorWidth * 1.0 / config.innerMostMBlock +
                (maxVectorWidth - config.innerMostKBlock % maxVectorWidth) %
                    maxVectorWidth * 1.0 / config.innerMostKBlock +
                (maxVectorWidth - config.innerMostNBlock % maxVectorWidth) %
                    maxVectorWidth * 1.0 / config.innerMostNBlock;
  return cost;
}

// calculate the cost of the workload balance
double workloadBalancedCost(linalg::LinalgOp &linalgOp,
                            CostModelOption option) {
  MatmulConfig config = option.cfg;
  assert(option.shape.size() >= 3 && "shape.size() should >= 3");
  uint32_t M = option.shape[0], N = option.shape[1], K = option.shape[2];
  uint32_t MTaskNum = llvm::divideCeil(M, config.MBlock);
  uint32_t NTaskNum = llvm::divideCeil(N, config.NBlock);
  uint32_t KTaskNum = llvm::divideCeil(K, config.KBlock);
  uint32_t actualThreads =
      llvm::divideCeil(MTaskNum, llvm::divideCeil(MTaskNum, config.MThreads)) *
      llvm::divideCeil(NTaskNum, llvm::divideCeil(NTaskNum, config.NThreads)) *
      llvm::divideCeil(KTaskNum, llvm::divideCeil(KTaskNum, config.KThreads));
  double cost = option.sysDesc.getNumThreads() * 1.0 / actualThreads;
  if (option.verbose) {
    LLVM_DEBUG(llvm::dbgs() << "MTaskNum is: " << MTaskNum << "\n");

    LLVM_DEBUG(llvm::dbgs()
               << "MThreads usage is: "
               << llvm::divideCeil(MTaskNum,
                                   llvm::divideCeil(MTaskNum, config.MThreads))
               << "\n");
    LLVM_DEBUG(llvm::dbgs()
               << "NThreads usage is: "
               << llvm::divideCeil(NTaskNum,
                                   llvm::divideCeil(NTaskNum, config.NThreads))
               << "\n");
    LLVM_DEBUG(llvm::dbgs()
               << "KThreads usage is: "
               << llvm::divideCeil(KTaskNum,
                                   llvm::divideCeil(KTaskNum, config.KThreads))
               << "\n");
    LLVM_DEBUG(llvm::dbgs()
               << "actualThreads usage is: " << actualThreads << "\n");
  }
  return cost;
}

// calculate the cost of the memory consumption on the thread
double memoryConsumptionOnThreadCost(linalg::LinalgOp &linalgOp,
                                     CostModelOption option) {
  assert(option.shape.size() >= 3 && "shape.size() should >= 3");
  MatmulConfig config = option.cfg;
  uint32_t M = option.shape[0], N = option.shape[1], K = option.shape[2];
  size_t dtypeSize = DataLayout().getTypeSize(
      ShapeAdaptor(linalgOp.getDpsInputs()[1].getType()).getElementType());
  // if use K split, there will be one more final reduce and break the post
  // fusion
  double KSplitPenalty = 8.0 * dtypeSize;
  double memoryConsumptionPerThread =
      M * K * 1.0 / config.MThreads / config.KThreads +
      K * N * 1.0 / config.KThreads / config.NThreads +
      M * N * ((config.KThreads - 1) * KSplitPenalty + 1.0) / config.MThreads /
          config.NThreads;
  return memoryConsumptionPerThread;
}

// calculate the cost of the computation intensity on the L2 cache
double computationIntensityOnL2Cache(linalg::LinalgOp &linalgOp,
                                     CostModelOption option) {
  MatmulConfig config = option.cfg;
  double fullLoadRatio = 0.7;
  uint32_t L2Cache = option.sysDesc.getCacheSize(2);
  size_t dtypeSize = DataLayout().getTypeSize(
      ShapeAdaptor(linalgOp.getDpsInputs()[1].getType()).getElementType());
  uint32_t outOfCachePenalty = 1024;
  double FLOPS = 2.0 * config.MBlock * config.NBlock * config.KBlock;
  double memoryConsumption = config.MBlock * config.NBlock +
                             config.NBlock * config.KBlock +
                             config.MBlock * config.KBlock;
  double computationIntensity = FLOPS / memoryConsumption;
  if (memoryConsumption * dtypeSize > L2Cache * fullLoadRatio)
    computationIntensity /= outOfCachePenalty;
  return 1 / computationIntensity;
}

// Bufferization may insert more memref.copy/brgemm cannot verify sucessfully
// and fall back to linalg lower if the buffer is dynamic
// Bufferization may insert more memref.copy/brgemm cannot verify sucessfully
// and fall back to linalg lower if the buffer is dynamic
double dynamicBufferizationCost(linalg::LinalgOp &linalgOp,
                                CostModelOption option) {
  auto config = option.cfg;
  assert(option.shape.size() >= 3 && "shape.size() should >= 3");
  uint32_t M = option.shape[0], N = option.shape[1];

  double cost = 0;
  uint32_t MNumBlockPerThread =
      llvm::divideCeil(M / config.innerMostMBlock, config.MThreads);
  uint32_t MNumInnerBlockPerBlock =
      llvm::divideCeil(config.MBlock, config.innerMostMBlock);
  assert(MNumInnerBlockPerBlock > 0 && "Invalid MNumInnerBlockPerBlock.");
  uint32_t MCost = MNumBlockPerThread % MNumInnerBlockPerBlock != 0 ||
                   (M / config.innerMostNBlock % config.MThreads != 0 &&
                    config.MBlock != config.innerMostMBlock);
  uint32_t NNumBlockPerThread =
      llvm::divideCeil(N / config.innerMostNBlock, config.NThreads);
  uint32_t NNumInnerBlockPerBlock =
      llvm::divideCeil(config.NBlock, config.innerMostNBlock);
  assert(NNumInnerBlockPerBlock > 0 && "Invalid NNumInnerBlockPerBlock.");
  uint32_t NCost = NNumBlockPerThread % NNumInnerBlockPerBlock != 0 ||
                   (N / config.innerMostNBlock % config.NThreads != 0 &&
                    config.NBlock != config.innerMostNBlock);
  cost = MCost + NCost;
  return cost;
}

double KSlicingCost(linalg::LinalgOp &linalgOp, CostModelOption option) {
  assert(option.shape.size() >= 3 && "shape.size() should >= 3");
  MatmulConfig config = option.cfg;
  double cost = config.KThreads;
  const unsigned int bigKThreshold = 4096;
  if (option.shape[2] >= bigKThreshold) {
    cost = config.KThreads <= 2 ? 1 : config.KThreads;
  }
  return cost;
}

double CBufferResuseCost(linalg::LinalgOp &linalgOp, CostModelOption option) {
  MatmulConfig config = option.cfg;
  return config.innerMostKBlock * 1.0 / config.KBlock;
}

double paddingCost(linalg::LinalgOp &linalgOp, CostModelOption option) {
  MatmulConfig config = option.cfg;
  double cost = 0;
  uint32_t M = option.shape[0], N = option.shape[1], K = option.shape[2];
  bool isPadOnM = M % config.innerMostMBlock != 0,
       isPadOnK = K % config.innerMostKBlock != 0,
       isPadOnN = N % config.innerMostNBlock != 0;
  if (isPadOnM || isPadOnK) {
    cost += llvm::divideCeil(M, config.innerMostMBlock) *
            llvm::divideCeil(K, config.innerMostKBlock);
  }
  if (isPadOnK || isPadOnN) {
    cost += llvm::divideCeil(N, config.innerMostNBlock) *
            llvm::divideCeil(K, config.innerMostKBlock);
  }
  if (isPadOnM || isPadOnN) {
    cost += llvm::divideCeil(N, config.innerMostNBlock) *
            llvm::divideCeil(M, config.innerMostMBlock);
  }
  return cost;
}

// filter the config by the cost model
std::vector<MatmulConfig>
filterConfigByCostModel(ArrayRef<MatmulConfig> configs,
                        linalg::LinalgOp &linalgOp, ArrayRef<uint32_t> shape,
                        CPUTargetDescriptionAnalysis &sysDesc,
                        const CostModelFn &costModel, float preserveRatio = 0.5,
                        float threshold = -1) {
  std::vector<MatmulConfig> result;
  std::vector<float> costs;
  std::vector<size_t> idx;
  for (auto &&[i, config] : llvm::enumerate(configs)) {
    costs.push_back(costModel(linalgOp, {shape, config, sysDesc, false}));
    idx.push_back(i);
  }
  std::stable_sort(idx.begin(), idx.end(), [&costs](size_t i1, size_t i2) {
    return costs[i1] < costs[i2];
  });
  double thresholdCost = costs[idx[(size_t)(preserveRatio * configs.size())]];
  thresholdCost =
      threshold < thresholdCost && threshold > 0 ? threshold : thresholdCost;
  for (const auto &i : idx)
    if (costs[i] <= thresholdCost)
      result.push_back(configs[i]);

  LLVM_DEBUG(llvm::dbgs() << "thresholdCost is: " << thresholdCost
                          << "\nbest with cost: " << costs[idx[0]] << "\n"
                          << configs[idx[0]] << "\n worst with cost: "
                          << costs[idx[configs.size() - 1]] << "\n"
                          << configs[idx[configs.size() - 1]] << "\n");
  if (result.empty())
    result = configs;
  return result;
}

// prepare the config candidates
std::vector<MatmulConfig>
prepareConfigCandidates(Operation *root, CPUTargetDescriptionAnalysis &sysDesc,
                        ArrayRef<uint32_t> shape,
                        ArrayRef<uint32_t> givenInnermostBlock,
                        bool allowIndivisibleInnerblock = false) {
  LLVM_DEBUG(llvm::dbgs() << "allowIndivisibleInnerblock: "
                          << allowIndivisibleInnerblock << "\n");
  assert(shape.size() >= 3 && "shape.size() should >= 3");
  std::vector<MatmulConfig> configs;
  uint32_t threads = sysDesc.getNumThreads();
  std::vector<uint32_t> MThreadsCandidates =
      getCandidate((uint32_t)threads, 1U);
  std::vector<uint32_t> NThreadsCandidates =
      getCandidate((uint32_t)threads, 1U);
  std::vector<uint32_t> KThreadsCandidates =
      getCandidate((uint32_t)threads, 1U);
  uint32_t noSmallBlockNeedThreshold = 8 * 4U;
  std::vector<uint32_t> MBlockCandidates = getCandidate(
      (uint32_t)shape[0], shape[0] >= noSmallBlockNeedThreshold ? 8U : 1U,
      (uint32_t)shape[0]);
  std::vector<uint32_t> NBlockCandidates =
      getCandidate((uint32_t)shape[1],
                   shape[1] >= noSmallBlockNeedThreshold ? 8U : 1U, shape[1]);
  std::vector<uint32_t> KBlockCandidates =
      getCandidate((uint32_t)shape[2],
                   shape[2] >= noSmallBlockNeedThreshold ? 8U : 1U, shape[2]);
  std::vector<uint32_t> innerMostMBlockCandidates =
      givenInnermostBlock[0] != 0 && givenInnermostBlock.size() == 3
          ? std::vector<uint32_t>{givenInnermostBlock[0]}
          : getCandidate((uint32_t)shape[0],
                         shape[0] >= noSmallBlockNeedThreshold ? 8U : 1U, 256U);
  std::vector<uint32_t> innerMostNBlockCandidates =
      givenInnermostBlock[1] != 0 && givenInnermostBlock.size() == 3
          ? std::vector<uint32_t>{givenInnermostBlock[1]}
          : getCandidate((uint32_t)shape[1],
                         shape[1] >= noSmallBlockNeedThreshold ? 8U : 1U, 256U);
  std::vector<uint32_t> innerMostKBlockCandidates =
      givenInnermostBlock[2] != 0 && givenInnermostBlock.size() == 3
          ? std::vector<uint32_t>{givenInnermostBlock[2]}
          : getCandidate((uint32_t)shape[2],
                         shape[2] >= noSmallBlockNeedThreshold ? 8U : 1U, 256U);

  if (allowIndivisibleInnerblock) {
    if (getElementTypeOrSelf(root->getOperandTypes()[1]).isBF16()) {
      innerMostKBlockCandidates = shape[2] < 32 ? std::vector<uint32_t>{16, 32}
                                                : std::vector<uint32_t>{32};
      innerMostNBlockCandidates = shape[1] < 32 ? std::vector<uint32_t>{16, 32}
                                                : std::vector<uint32_t>{32};
    } else {
      innerMostKBlockCandidates = std::vector<uint32_t>{16, 32, 64};
      innerMostNBlockCandidates = std::vector<uint32_t>{16, 32, 64};
    }

    NBlockCandidates = innerMostNBlockCandidates;
    KBlockCandidates = innerMostKBlockCandidates;
  }

  bool isVNNIMM2D =
      linalgx::isGenericPackedMatmulOp(root, linalgx::PackingType::VNNI_MM2D);
  // TODO: improve via multi threading or add more constraints to restrict
  // the candidate size
  for (uint32_t MThreads : MThreadsCandidates) {
    for (uint32_t NThreads : NThreadsCandidates) {
      for (uint32_t KThreads : KThreadsCandidates) {
        if (!validateThreads({MThreads, NThreads, KThreads}, sysDesc))
          continue;
        for (uint32_t MBlock : MBlockCandidates) {
          for (uint32_t innerMostMBlock : innerMostMBlockCandidates) {
            for (uint32_t NBlock : NBlockCandidates) {
              for (uint32_t innerMostNBlock : innerMostNBlockCandidates) {
                for (uint32_t KBlock : KBlockCandidates) {
                  for (uint32_t innerMostKBlock : innerMostKBlockCandidates) {
                    MatmulConfig config{
                        MThreads,        NThreads,        KThreads,
                        MBlock,          NBlock,          KBlock,
                        innerMostMBlock, innerMostNBlock, innerMostKBlock};
                    if (validateConfig(config, shape,
                                       allowIndivisibleInnerblock, isVNNIMM2D))
                      configs.push_back(config);
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  LLVM_DEBUG(
      llvm::dbgs() << "Finish generating candidates. ConfigCandidates size: "
                   << configs.size() << "\n");
  return configs;
}

// read the config from the attributes for tuning
bool readConfigFromAttrs(MatmulConfig &config, ArrayRef<NamedAttribute> attrs) {
  size_t cfgItemCnt = 0;
  for (const auto &attr : attrs) {
    if (attr.getName() == "KBlock") {
      config.KBlock = cast<IntegerAttr>(attr.getValue()).getInt();
      cfgItemCnt++;
    } else if (attr.getName() == "KThreads") {
      config.KThreads = cast<IntegerAttr>(attr.getValue()).getInt();
      cfgItemCnt++;
    } else if (attr.getName() == "NBlock") {
      config.NBlock = cast<IntegerAttr>(attr.getValue()).getInt();
      cfgItemCnt++;
    } else if (attr.getName() == "NThreads") {
      config.NThreads = cast<IntegerAttr>(attr.getValue()).getInt();
      cfgItemCnt++;
    } else if (attr.getName() == "MBlock") {
      config.MBlock = cast<IntegerAttr>(attr.getValue()).getInt();
      cfgItemCnt++;
    } else if (attr.getName() == "MThreads") {
      config.MThreads = cast<IntegerAttr>(attr.getValue()).getInt();
      cfgItemCnt++;
    } else if (attr.getName() == "innermostMBlock") {
      config.innerMostMBlock = cast<IntegerAttr>(attr.getValue()).getInt();
      cfgItemCnt++;
    } else if (attr.getName() == "innermostNBlock") {
      config.innerMostNBlock = cast<IntegerAttr>(attr.getValue()).getInt();
      cfgItemCnt++;
    } else if (attr.getName() == "innermostKBlock") {
      config.innerMostKBlock = cast<IntegerAttr>(attr.getValue()).getInt();
      cfgItemCnt++;
    }
  }
  return cfgItemCnt == 9;
}

bool readAndValidateConfig(MatmulConfig &config,
                           const linalg::LinalgOp &linalgOp,
                           ArrayRef<uint32_t> shape,
                           bool allowIndivisibleInnerBlock) {
  SmallVector<NamedAttribute> attrs(linalgOp->getAttrs());
  bool fullConfig = readConfigFromAttrs(config, attrs);
  if (!fullConfig) {
    LLVM_DEBUG(llvm::dbgs() << "Missing fields in predefined config.\n");
    return false;
  }
  bool validConfig =
      validateConfig(config, shape, allowIndivisibleInnerBlock,
                     linalgx::isGenericPackedMatmulOp(
                         linalgOp, linalgx::PackingType::VNNI_MM2D));
  if (!validConfig) {
    LLVM_DEBUG(llvm::dbgs() << "Invalid predefined config.\n");
    return false;
  }
  return true;
}

// Analyze the workload and system description to generate the default config
// Factor to consider:
// thread utilization
// computation intensity
// cache locality
// memory requirements
// computation unit efficiency
// padding/pack cost
// workload balance
// communication
// previous matmul
// C buffer reuse
MatmulConfig MatmulConfigAnalysis::getConfig() {
  if (!hasConfig) {
    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(root)) {
      CPUTargetDescriptionAnalysis sysDesc(root);
      SmallVector<SmallVector<DimType>> oprandDimType =
          *getOprandDimType(linalgOp);
      // get the origin M,N,K size
      SmallVector<unsigned> MDimTypeIdx =
          extractDimTypeIdx(oprandDimType[0], DimType::M);
      SmallVector<unsigned> KDimTypeIdx =
          extractDimTypeIdx(oprandDimType[1], DimType::K);
      SmallVector<unsigned> NDimTypeIdx =
          extractDimTypeIdx(oprandDimType[1], DimType::N);
      uint32_t M = 1U, N = 1U, K = 1U;
      for (auto &&[s, dimType] :
           llvm::zip(linalgOp.getShape(linalgOp.getDpsInputOperand(0)),
                     oprandDimType[0]))
        if (dimType == DimType::M)
          M *= s;
      for (auto &&[s, dimType] :
           llvm::zip(linalgOp.getShape(linalgOp.getDpsInputOperand(1)),
                     oprandDimType[1])) {
        if (dimType == DimType::N)
          N *= s;
        else if (dimType == DimType::K)
          K *= s;
      }

      // innermost Block, if the layout is blocked layout, the innermost block
      // will derived from the layout directly
      uint32_t defaultBlock = 32;
      config.innerMostMBlock = M % defaultBlock == 0 ? defaultBlock : M;
      config.innerMostNBlock = N % defaultBlock == 0 ? defaultBlock : N;
      config.innerMostKBlock = K % defaultBlock == 0 ? defaultBlock : K;
      SmallVector<uint32_t> givenInnermostBlock;
      if (MDimTypeIdx.size() > 1) {
        config.innerMostMBlock = 1;
        for (auto &&[i, d] : llvm::enumerate(MDimTypeIdx))
          if (i != 0)
            config.innerMostMBlock *=
                linalgOp.getShape(linalgOp.getDpsInputOperand(0))[d];
        givenInnermostBlock.push_back(config.innerMostMBlock);
      } else {
        givenInnermostBlock.push_back(0);
      }
      if (NDimTypeIdx.size() > 1) {
        config.innerMostNBlock = 1;
        for (auto &&[i, d] : llvm::enumerate(NDimTypeIdx))
          if (i != 0)
            config.innerMostNBlock *=
                linalgOp.getShape(linalgOp.getDpsInputOperand(1))[d];
        givenInnermostBlock.push_back(config.innerMostNBlock);
      } else {
        givenInnermostBlock.push_back(0);
      }
      if (KDimTypeIdx.size() > 1) {
        config.innerMostKBlock = 1;
        for (auto &&[i, d] : llvm::enumerate(KDimTypeIdx))
          if (i != 0)
            config.innerMostKBlock *=
                linalgOp.getShape(linalgOp.getDpsInputOperand(1))[d];
        givenInnermostBlock.push_back(config.innerMostKBlock);
      } else {
        givenInnermostBlock.push_back(0);
      }

      LLVM_DEBUG(llvm::dbgs()
                 << "M: " << M << ", N: " << N << ", K: " << K << "\n");

      // try to read the config from the attributes
      bool hasValidPredefinedConfig = readAndValidateConfig(
          config, linalgOp, SmallVector<uint32_t>{M, N, K},
          allowIndivisibleInnerBlock);

      SmallVector<std::tuple<CostModelFn, std::string, double>> costModelList =
          {{dynamicBufferizationCost, "dynamicBufferizationCost", 0},
           {CBufferResuseCost, "CBufferResuseCost", -1},
           {workloadBalancedCost, "workloadBalancedCost", 1.9},
           {vectorRegEfficiencyCost, "vectorRegEfficiencyCost ", -1},
           {computationIntensityOnL2Cache, "computationIntensityOnL2Cache", -1},
           {KSlicingCost, "KSlicingCost", -1},
           {memoryConsumptionOnThreadCost, "memoryConsumptionOnThreadCost", -1},
           {paddingCost, "paddingCost", -1}};
      SmallVector<uint32_t> shape = {M, N, K};

      // if there is a given config, skip the cost model
      if (!hasValidPredefinedConfig) {
        LLVM_DEBUG(
            llvm::dbgs()
            << "No valid predefined config. Setting with default config.\n");
        // TODO: Could add a weight or priority for cost model
        std::vector<MatmulConfig> configCandidates =
            prepareConfigCandidates(root, sysDesc, shape, givenInnermostBlock,
                                    allowIndivisibleInnerBlock);
        for (auto &&[fn, name, threshold] : costModelList) {
          LLVM_DEBUG(llvm::dbgs() << name << "\n");
          configCandidates = filterConfigByCostModel(
              configCandidates, linalgOp, shape, sysDesc, fn, 0.5, threshold);
        }
        if (!configCandidates.empty())
          config = configCandidates[0];

        assert(validateConfig(config, shape, allowIndivisibleInnerBlock,
                              linalgx::isGenericPackedMatmulOp(
                                  root, linalgx::PackingType::VNNI_MM2D)) &&
               "config is invalid");
      }

      LLVM_DEBUG(llvm::dbgs()
                 << "Final config\nNumThreads: " << sysDesc.getNumThreads()
                 << ", MatmulConfig: " << config << "\n");
      for (auto &&[fn, name, threshold] : costModelList) {
        LLVM_DEBUG(llvm::dbgs()
                   << name << ": "
                   << fn(linalgOp, {shape, config, sysDesc, true}) << "\n");
      }
    }
    hasConfig = true;
  }

  return config;
}
} // namespace gc
} // namespace mlir