//===-- MatmulConfigAnalysis.cpp - DESC -------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <memory>

#include "gc/Analysis/MatmulConfigAnalysis.h"
// #include "json/json.h"
#include <fstream>
namespace mlir {
namespace gc {

#define DEBUG_TYPE "matmul-config-analysis"

#define MAX_THREADS (1024U * 1024U)

llvm::raw_ostream &operator<<(llvm::raw_ostream &ss,
                              const MatmulConfig &config) {

  ss << "MBlock: " << config.MBlock << ", NBlock: " << config.NBlock
     << ", KBlock: " << config.KBlock << ", MThreads: " << config.MThreads
     << ", NThreads: " << config.NThreads << ", KThreads: " << config.KThreads
     << ", innerMostMBlock: " << config.innerMostMBlock
     << ", innerMostNBlock: " << config.innerMostNBlock
     << ", innerMostKBlock: " << config.innerMostKBlock;
  return ss;
}

std::vector<uint32_t> getCandidate(uint32_t num, uint32_t floor,
                                   uint32_t ceil) {
  // factor
  std::vector<uint32_t> candidates;
  for (uint32_t i = 1; i <= num; i++) {
    if (num % i == 0 && i <= ceil && i >= floor) {
      candidates.push_back(i);
    }
  }
  // the pow of 2
  auto candidate = 1U;
  while (candidate < num && candidate <= ceil && candidate >= floor) {
    candidates.push_back(candidate);
    candidate *= 2;
  }
  auto last = std::unique(candidates.begin(), candidates.end());
  candidates.erase(last, candidates.end());
  return candidates;
}

bool isValidConfig(const MatmulConfig &config, SystemDesc &sysDesc,
                   ArrayRef<uint32_t> shape) {
  if (config.innerMostMBlock == 0 || config.innerMostNBlock == 0 ||
      config.innerMostKBlock == 0) {
    return false;
  }
  if (config.MBlock % config.innerMostMBlock != 0 ||
      config.NBlock % config.innerMostNBlock != 0 ||
      config.KBlock % config.innerMostKBlock != 0) {
    return false;
  }
  auto threads = sysDesc.getNumThreads();
  if (config.MThreads * config.NThreads * config.KThreads != threads) {
    return false;
  }

  if (shape[0] % config.innerMostMBlock != 0 ||
      shape[1] % config.innerMostNBlock != 0 ||
      shape[2] % config.innerMostKBlock != 0) {
    return false;
  }

  return true;
}

double hardwareEfficiencyCost(linalg::LinalgOp &linalgOp,
                              ArrayRef<uint32_t> shape,
                              const MatmulConfig &config, SystemDesc &sysDesc) {
  auto dtypeSize = DataLayout().getTypeSizeInBits(
      ShapeAdaptor(linalgOp.getDpsInputs()[1].getType()).getElementType());
  auto vectorLength = sysDesc.getContractionOperationMaxVectorLength();
  auto mMaxVectorLength = vectorLength[0] / dtypeSize;
  auto kMaxVectorLength =
      (vectorLength.size() > 1 ? vectorLength[1] : vectorLength[0]) / dtypeSize;
  auto cost = (mMaxVectorLength - config.innerMostMBlock % mMaxVectorLength) %
                  mMaxVectorLength * 1.0 / config.innerMostMBlock +
              (kMaxVectorLength - config.innerMostKBlock % kMaxVectorLength) %
                  kMaxVectorLength * 1.0 / config.innerMostKBlock +
              (mMaxVectorLength - config.innerMostNBlock % mMaxVectorLength) %
                  mMaxVectorLength * 1.0 / config.innerMostNBlock;
  return cost;
}

double workloadBalancedCost(linalg::LinalgOp &linalgOp,
                            ArrayRef<uint32_t> shape,
                            const MatmulConfig &config, SystemDesc &sysDesc) {
  auto M = shape[0], N = shape[1], K = shape[2];
  auto MTaskNum = llvm::divideCeil(M, config.MBlock);
  auto NTaskNum = llvm::divideCeil(N, config.NBlock);
  auto KTaskNum = llvm::divideCeil(K, config.KBlock);
  auto cost = (MTaskNum % config.MThreads) * 1.0 / MTaskNum +
              (NTaskNum % config.NThreads) * 1.0 / NTaskNum +
              (KTaskNum % config.KThreads) * 1.0 / KTaskNum;
  if(MTaskNum < config.MThreads || NTaskNum < config.NThreads || KTaskNum < config.KThreads)
  {
    auto threadNotFulllyUtilizedPenalty = 10.0;
    cost *= threadNotFulllyUtilizedPenalty;
  }
  return cost;
}
constexpr unsigned bitPerByte = 8;
double memoryConsumptionOnThreadCost(linalg::LinalgOp &linalgOp,
                                     ArrayRef<uint32_t> shape,
                                     const MatmulConfig &config,
                                     SystemDesc &sysDesc) {
  auto M = shape[0], N = shape[1], K = shape[2];
  auto dtypeSize = DataLayout().getTypeSizeInBits(
      ShapeAdaptor(linalgOp.getDpsInputs()[1].getType()).getElementType());
  // if use K split, there will be one more final reduce and break the post
  // fusion

  auto KSplitPenalty = 8.0 * (dtypeSize / bitPerByte);
  auto memoryConsumptionPerThread =
      M * K * 1.0 / config.MThreads / config.KThreads +
      K * N * 1.0 / config.KThreads / config.NThreads +
      M * N * ((config.KThreads - 1) * KSplitPenalty + 1.0) / config.MThreads /
          config.NThreads;
  return memoryConsumptionPerThread;
}

double computationIntensityOnL2Cache(linalg::LinalgOp &linalgOp,
                                     ArrayRef<uint32_t> shape,
                                     const MatmulConfig &config,
                                     SystemDesc &sysDesc) {
  double simulationPenalty = 0.7;
  auto L2Cache = sysDesc.getCacheSize(2);
  auto dtypeSize = DataLayout().getTypeSizeInBits(
      ShapeAdaptor(linalgOp.getDpsInputs()[1].getType()).getElementType());
  auto outOfCachePenalty = 1024;
  double FLOPS = 2.0 * config.MBlock * config.NBlock * config.KBlock;
  double memoryConsumption = config.MBlock * config.NBlock +
                             config.NBlock * config.KBlock +
                             config.MBlock * config.KBlock;
  double computationIntensity = FLOPS / memoryConsumption;
  if (memoryConsumption * (dtypeSize / bitPerByte) > L2Cache * simulationPenalty)
  {
      computationIntensity /= outOfCachePenalty;
  }
  return 1 / computationIntensity;
}

using CostModelFn =
    std::function<float(linalg::LinalgOp &linalgOp, ArrayRef<uint32_t> shape,
                        MatmulConfig cfg, SystemDesc &sysDesc)>;

std::vector<MatmulConfig>
filterConfigByCostModel(std::vector<MatmulConfig> configs,
                        linalg::LinalgOp &linalgOp, ArrayRef<uint32_t> shape,
                        SystemDesc &sysDesc, CostModelFn costModel,
                        float eliminationRatio = 0.5, float threshold = -1) {
  std::vector<MatmulConfig> result;
  std::vector<float> costs;
  std::vector<size_t> idx;
  for (auto [i, config] : llvm::enumerate(configs)) {
    costs.push_back(costModel(linalgOp, shape, config, sysDesc));
    idx.push_back(i);
  }
  std::stable_sort(idx.begin(), idx.end(), [&costs](size_t i1, size_t i2) {
    return costs[i1] < costs[i2];
  });
  auto thresholdCost = costs[idx[(size_t)(eliminationRatio * configs.size())]];
  thresholdCost =
      threshold < thresholdCost && threshold > 0 ? threshold : thresholdCost;
  for (size_t i = 0; i < configs.size(); i++) {
    if (costs[idx[i]] <= thresholdCost) {
      result.push_back(configs[idx[i]]);
    }
  }
  llvm::outs() << "thresholdCost is: " << thresholdCost
               << "\nbest with cost: " << costs[idx[0]] << "\n"
               << configs[idx[0]]
               << "\n worst with cost: " << costs[idx[configs.size() - 1]]
               << "\n"
               << configs[idx[configs.size() - 1]] << "\n";
  return result.size() > 0 ? result : configs;
}

std::vector<MatmulConfig>
prepareConfigCandidates(Operation *root, SystemDesc &sysDesc,
                        ArrayRef<uint32_t> shape,
                        ArrayRef<uint32_t> givenInnermostBlock) {
  std::vector<MatmulConfig> configs;
  auto threads = sysDesc.getNumThreads();
  auto MThreadsCandidates = getCandidate((uint32_t)threads, 1U, MAX_THREADS);
  auto NThreadsCandidates = getCandidate((uint32_t)threads, 1U, MAX_THREADS);
  auto KThreadsCandidates = getCandidate((uint32_t)threads, 1U, MAX_THREADS);
  auto MBlockCandidates =
      getCandidate((uint32_t)shape[0], 1U, (uint32_t)shape[0]);
  auto NBlockCandidates = getCandidate((uint32_t)shape[1], 1U, shape[1]);
  auto KBlockCandidates = getCandidate((uint32_t)shape[2], 1U, shape[2]);
  auto innerMostMBlockCandidates =
      getCandidate((uint32_t)shape[0], 1U, (uint32_t)shape[0]);
  auto innerMostNBlockCandidates =
      getCandidate((uint32_t)shape[1], 1U, (uint32_t)shape[1]);
  auto innerMostKBlockCandidates =
      getCandidate((uint32_t)shape[2], 1U, (uint32_t)shape[2]);
  if (givenInnermostBlock.size() == 3) {
    innerMostMBlockCandidates =
        givenInnermostBlock[0] != 0
            ? std::vector<uint32_t>{givenInnermostBlock[0]}
            : innerMostMBlockCandidates;
    innerMostNBlockCandidates =
        givenInnermostBlock[1] != 0
            ? std::vector<uint32_t>{givenInnermostBlock[1]}
            : innerMostNBlockCandidates;
    innerMostKBlockCandidates =
        givenInnermostBlock[2] != 0
            ? std::vector<uint32_t>{givenInnermostBlock[2]}
            : innerMostKBlockCandidates;
  }
  llvm::outs() << "MThreadsCandidates size: " << MThreadsCandidates.size()
               << "\n";
  llvm::outs() << "NThreadsCandidates size: " << NThreadsCandidates.size()
               << "\n";
  llvm::outs() << "KThreadsCandidates size: " << KThreadsCandidates.size()
               << "\n";
  llvm::outs() << "MBlockCandidates size: " << MBlockCandidates.size() << "\n";
  llvm::outs() << "NBlockCandidates size: " << NBlockCandidates.size() << "\n";
  llvm::outs() << "KBlockCandidates size: " << KBlockCandidates.size() << "\n";
  llvm::outs() << "innerMostMBlockCandidates size: "
               << innerMostMBlockCandidates.size() << "\n";
  llvm::outs() << "innerMostNBlockCandidates size: "
               << innerMostNBlockCandidates.size() << "\n";
  llvm::outs() << "innerMostKBlockCandidates size: "
               << innerMostKBlockCandidates.size() << "\n";
  for (auto MThreads : MThreadsCandidates) {
    for (auto NThreads : NThreadsCandidates) {
      for (auto KThreads : KThreadsCandidates) {
        for (auto MBlock : MBlockCandidates) {
          for (auto NBlock : NBlockCandidates) {
            for (auto KBlock : KBlockCandidates) {
              for (auto innerMostMBlock : innerMostMBlockCandidates) {
                for (auto innerMostNBlock : innerMostNBlockCandidates) {
                  for (auto innerMostKBlock : innerMostKBlockCandidates) {
                    MatmulConfig config{
                        MBlock,          NBlock,          KBlock,
                        MThreads,        NThreads,        KThreads,
                        innerMostMBlock, innerMostNBlock, innerMostKBlock};

                    if (isValidConfig(config, sysDesc, shape)) {
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
  }
  return configs;
}

/*
thread utilization
computation intensity
cache locality
memory requirements
computation unit efficiency
padding/pack cost
workload balance
communication
previous matmul
*/
MatmulConfigAnalysis::MatmulConfigAnalysis(Operation *root) {
  SystemDesc sysDesc;
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(root)) {
      // TODO: build a more complex heuristic to determine the best tiling
      auto oprandDimType = *getOprandDimType(linalgOp);
      // get the origin M,N,K size
      auto MDimTypeIdx = extractDimTypeIdx(oprandDimType[0], DimType::M);
      auto KDimTypeIdx = extractDimTypeIdx(oprandDimType[1], DimType::K);
      auto NDimTypeIdx = extractDimTypeIdx(oprandDimType[1], DimType::N);
      uint32_t M = 1U, N = 1U, K = 1U;
      for (auto [s, dimType] :
           llvm::zip(linalgOp.getShape(linalgOp.getDpsInputOperand(0)),
                     oprandDimType[0]))
      {
          if (dimType == DimType::M)
          {
              M *= s;
          }
    }
    for (auto [s, dimType] :
         llvm::zip(linalgOp.getShape(linalgOp.getDpsInputOperand(1)),
                   oprandDimType[1])) {
      if (dimType == DimType::N) {
        N *= s;
      } else if (dimType == DimType::K) {
        K *= s;
      }
    }
    // innermost Block, if the layout is blockied layout, the innermost block
    // will derived from the layout directly
    auto defaultBlock = 32;
    config.innerMostMBlock = M % defaultBlock == 0 ? defaultBlock : M;
    config.innerMostNBlock = N % defaultBlock == 0 ? defaultBlock : N;
    config.innerMostKBlock = K % defaultBlock == 0 ? defaultBlock : K;
    SmallVector<uint32_t> givenInnermostBlock;
    if (MDimTypeIdx.size() > 1) {
      config.innerMostMBlock = 1;
      for (auto i = 1UL; i < MDimTypeIdx.size(); i++) {
        config.innerMostMBlock *=
            linalgOp.getShape(linalgOp.getDpsInputOperand(0))[MDimTypeIdx[i]];
      }
      givenInnermostBlock.push_back(config.innerMostMBlock);
    } else {
      givenInnermostBlock.push_back(0);
    }
    if (NDimTypeIdx.size() > 1) {
      config.innerMostNBlock = 1;
      for (auto i = 1UL; i < NDimTypeIdx.size(); i++) {
        config.innerMostNBlock *=
            linalgOp.getShape(linalgOp.getDpsInputOperand(1))[NDimTypeIdx[i]];
      }
      givenInnermostBlock.push_back(config.innerMostNBlock);
    } else {
      givenInnermostBlock.push_back(0);
    }
    if (KDimTypeIdx.size() > 1) {
      config.innerMostKBlock = 1;
      for (auto i = 1UL; i < KDimTypeIdx.size(); i++) {
        config.innerMostKBlock *=
            linalgOp.getShape(linalgOp.getDpsInputOperand(1))[KDimTypeIdx[i]];
      }
      givenInnermostBlock.push_back(config.innerMostKBlock);
    } else {
      givenInnermostBlock.push_back(0);
    }
    // Number of block
    auto MNumBlock = M / config.innerMostMBlock;
    auto NNumBlock = N / config.innerMostNBlock;
    auto KNumBlock = K / config.innerMostKBlock;
    // Threads
    config.MThreads = 32;
    config.NThreads = 1;
    config.KThreads = 1;

    // Block
    config.MBlock = (int)llvm::divideCeil(MNumBlock, config.MThreads) *
                    config.innerMostMBlock;
    config.NBlock = (int)llvm::divideCeil(NNumBlock, config.NThreads) *
                    config.innerMostNBlock;
    config.KBlock = (int)llvm::divideCeil(KNumBlock, config.KThreads) *
                    config.innerMostKBlock;
    config.MBlock = 128;
    config.NBlock = 128;
    config.KBlock = 128;
    config.MThreads = 2;
    config.NThreads = 2;
    config.KThreads = 1;

    llvm::outs() << "M: " << M << ", N: " << N << ", K: " << K << "\n";

    SmallVector<std::tuple<CostModelFn, std::string, double>> costModelList = {
        {workloadBalancedCost, "workloadBalancedCost", 1},
        {hardwareEfficiencyCost, "hardwareEfficiencyCost", -1},
        {computationIntensityOnL2Cache, "computationIntensityOnL2Cache", -1},
        {memoryConsumptionOnThreadCost, "memoryConsumptionOnThreadCost", -1}};

    auto configCandidates =
        prepareConfigCandidates(root, sysDesc, {M, N, K}, givenInnermostBlock);
    for (auto [fn, name, threshold] : costModelList) {
        llvm::outs() << "\n" << name << "\n";
        configCandidates = filterConfigByCostModel(configCandidates, linalgOp,
                                                   {M, N, K}, sysDesc, fn, 0.5, threshold);
        llvm::outs() << "ConfigCandidates size: " << configCandidates.size()
                     << "\n";
    }

    if (configCandidates.size() > 0) {
      config = configCandidates[0];
    }

    // Json::Value cfg;
    // std::ifstream cfgFile("/home/zhicong/code/tpp-mlir-ext/build/cfg.json",
    //                       std::ifstream::binary);
    // if (cfgFile.is_open()) {
    //   cfgFile >> cfg;
    //   bool use = cfg["use"].asBool();
    //   if (use) {
    //     config.MBlock = cfg["MBlock"].asUInt();
    //     config.NBlock = cfg["NBlock"].asUInt();
    //     config.KBlock = cfg["KBlock"].asUInt();
    //     config.MThreads = cfg["MThreads"].asUInt();
    //     config.NThreads = cfg["NThreads"].asUInt();
    //     config.KThreads = cfg["KThreads"].asUInt();
    //     config.innerMostMBlock = cfg["innerMostMBlock"].asUInt();
    //     config.innerMostNBlock = cfg["innerMostNBlock"].asUInt();
    //     config.innerMostKBlock = cfg["innerMostKBlock"].asUInt();
    //   }
    // }
    llvm::outs() << "Final config\nNumThreads: " << sysDesc.getNumThreads()
                 << ", MatmulConfig: " << config << "\n";
    for (auto [fn, name, threshold] : costModelList)
    {
        auto cost = fn(linalgOp,
                              {M, N, K}, config, sysDesc);
        llvm::outs() << name << ": " << cost
                     << "\n";
    }
  }
}
} // namespace gc
} // namespace mlir