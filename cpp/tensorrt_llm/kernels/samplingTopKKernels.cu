/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#elif (CUDART_VERSION >= 11050)
#include <cub/cub.cuh>
#else
#include "3rdparty/cub/cub.cuh"
#endif

#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include "tensorrt_llm/kernels/samplingTopKKernels.h"

using namespace tensorrt_llm::common;
using namespace tensorrt_llm::runtime;

namespace tensorrt_llm::kernels
{

template <typename T, int32_t BLOCK_SIZE_, int32_t BLOCKS_PER_BEAM_>
__global__ void topKStage1(T const* __restrict logProbs, T const* const* __restrict logProbsPtrs, T* tmpLogProbs,
    SizeType32* topKTmpIdBuf, T* topKTmpValBuf, FinishedState const* finished, SizeType32 maxTopK,
    SizeType32 const* topKs, SizeType32 vocabSize, TokenIdType const* endIds, bool const* skipDecode,
    SizeType32 const* batchSlots, SizeType32 const* tokensPerStep, SizeType32 maxTokensPerStep)
{
    typedef cub::BlockReduce<TopK_2<T>, BLOCK_SIZE_> BlockReduce;
    __shared__ typename BlockReduce::TempStorage tempStorage;

    auto const tid = static_cast<SizeType32>(threadIdx.x);
    auto const bid = static_cast<SizeType32>(blockIdx.x);
    auto const tokenIdx = static_cast<SizeType32>(blockIdx.y);

    auto const batchId = bid / BLOCKS_PER_BEAM_; // row id for logProbs
    auto const batchSlot = batchSlots == nullptr ? batchId : batchSlots[batchId];
    if (tokensPerStep != nullptr && tokenIdx >= tokensPerStep[batchSlot])
    {
        return;
    }

    FinishedState const finishState = finished != nullptr ? finished[batchSlot] : FinishedState::empty();
    if ((skipDecode != nullptr && skipDecode[batchSlot]) || (finishState.isSkipDecoding()))
    {
        return;
    }

    auto const logBufIndex = batchId * maxTokensPerStep * vocabSize + tokenIdx * vocabSize;
    auto logProbsSlot
        = logProbsPtrs == nullptr ? logProbs + logBufIndex : logProbsPtrs[batchId * maxTokensPerStep + tokenIdx];
    auto const blockLane = bid % BLOCKS_PER_BEAM_;                  // block id for a beam
    auto const k = (topKs != nullptr) ? topKs[batchSlot] : maxTopK; // batchId = batch index

    auto const tmpLogBufIndex = batchId * maxTokensPerStep * vocabSize + tokenIdx * vocabSize;
    auto const tmpTopKBufIndex = batchId * maxTokensPerStep * BLOCKS_PER_BEAM_ * maxTopK
        + tokenIdx * BLOCKS_PER_BEAM_ * maxTopK + blockLane * k;

    TopK_2<T> partial;
    bool const IS_FP16 = std::is_same<T, half>::value;
    T const MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;

    if (finished != nullptr && finishState.isFinished())
    {
        if (tid < k && endIds != nullptr) // if returnAllSelectedToken, endIds would not be an input
        {
            auto const index = tmpTopKBufIndex + tid;
            // endId=-1 means generation doesn't stop upon encountering a certain token.
            if (blockLane == 0 && tid == 0 && endIds[batchSlot] > -1)
            {
                auto const endId = endIds[batchSlot];
                topKTmpIdBuf[index] = tmpLogBufIndex + endId;
                topKTmpValBuf[index] = logProbsSlot[endId];
            }
            else
            {
                topKTmpIdBuf[index] = -1;
                topKTmpValBuf[index] = -MAX_T_VAL;
            }
        }
        return;
    }

    for (auto elemId = tid + blockLane * BLOCK_SIZE_; elemId < vocabSize; elemId += BLOCK_SIZE_ * BLOCKS_PER_BEAM_)
    {
        auto localIndex = elemId + tmpLogBufIndex;
        tmpLogProbs[localIndex] = logProbsSlot[elemId];
    }

    for (SizeType32 ite = 0; ite < k; ite++)
    {
        partial.init();
#pragma unroll
        for (auto elemId = tid + blockLane * BLOCK_SIZE_; elemId < vocabSize; elemId += BLOCK_SIZE_ * BLOCKS_PER_BEAM_)
        {
            auto index = elemId + tmpLogBufIndex;
            partial.insert(tmpLogProbs[index], index);
        }

        TopK_2<T> total = BlockReduce(tempStorage).Reduce(partial, reduce_topk_op_2<T>);

        if (tid == 0)
        {
            auto const index = tmpTopKBufIndex + ite;
            topKTmpIdBuf[index] = total.p;
            topKTmpValBuf[index] = total.u;
            if (total.p >= 0)
            {
                tmpLogProbs[total.p] = -MAX_T_VAL;
            }
        }
        __syncthreads();
    }
}

template <typename T, int BLOCK_SIZE_, int BLOCKS_PER_BEAM_>
__global__ void topKStage2Sampling(SizeType32 const* __restrict topKTmpIdBuf, T* topKTmpValBuf, TokenIdType** idsPtrs,
    TokenIdType* ids, SizeType32* sequenceLengths, FinishedState const* finishedInput, FinishedState* finishedOutput,
    float* cumLogProbs, float* outputLogProbs, SizeType32 maxTopK, SizeType32 const* topKs, float topP,
    float const* topPs, curandState_t* curandState, TokenIdType const* endIds, SizeType32 vocabSize,
    bool const* skipDecode, SizeType32 const* batchSlots, SizeType32 maxBatchSize, bool normalizeLogProbs,
    bool logitHasProbs, SizeType32 const* tokensPerStep, SizeType32 maxTokensPerStep, SizeType32 maxSeqLen,
    bool returnAllSelectedTokensFlag, bool strictTopPBoundary, bool const* returnAllSelectedTokensPerSlot,
    TokenIdType* outputIdCurrentStep, bool const* skipOutputIdCurrentStep)
{
    bool const IS_FP16 = std::is_same<T, half>::value;
    T const MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;

    auto const tid = static_cast<SizeType32>(threadIdx.x);
    auto const batchIdx = static_cast<SizeType32>(blockIdx.x);
    auto const tokenIdx = static_cast<SizeType32>(blockIdx.y);
    auto const batchSlot = batchSlots == nullptr ? batchIdx : batchSlots[batchIdx];
    
    FinishedState const finishState = finishedInput != nullptr ? finishedInput[batchSlot] : FinishedState::empty();
    if ((skipDecode != nullptr && skipDecode[batchSlot]) || (finishState.isSkipDecoding()))
    {
        return;
    }
    if (tokensPerStep != nullptr && tokenIdx >= tokensPerStep[batchSlot])
    {
        return;
    }

    auto const k = (topKs != nullptr) ? topKs[batchSlot] : maxTopK;
    auto const probThreshold = (topPs != nullptr) ? topPs[batchSlot] : topP;
    auto const size = k * BLOCKS_PER_BEAM_;
    auto const stride = maxTopK * BLOCKS_PER_BEAM_;
    auto const returnAllSelectedTokens = returnAllSelectedTokensPerSlot != nullptr
        ? returnAllSelectedTokensPerSlot[batchSlot]
        : returnAllSelectedTokensFlag;
    bool const sampleTokenInSelected = returnAllSelectedTokens && outputIdCurrentStep && curandState
        && skipOutputIdCurrentStep && !skipOutputIdCurrentStep[batchSlot];

    typedef cub::BlockReduce<TopK_2<float>, BLOCK_SIZE_> BlockReduce;
    __shared__ typename BlockReduce::TempStorage tempStorage;
    extern __shared__ char array[];
    __shared__ float sSum;
    __shared__ float sMaxLogit;
    
    T* sVal = topKTmpValBuf + (batchIdx * maxTokensPerStep + tokenIdx) * stride;
    auto* sId = reinterpret_cast<SizeType32*>(array);
    auto* sLogits = reinterpret_cast<float*>(sId + k);        // Store selected logits
    auto* sProbs = reinterpret_cast<float*>(sLogits + k);     // Store computed probabilities
    
    if (tid == 0)
    {
        sSum = 0.0f;
        sMaxLogit = -MAX_T_VAL;
    }
    
    TopK_2<float> partial;

    if (finishState.isFinished())
    {
        if (finishedOutput != nullptr)
        {
            finishedOutput[batchSlot] = finishState;
        }
        return;
    }

    // Phase 1: Select top-K indices and find max logit for numerical stability
    for (SizeType32 ite = 0; ite < k; ite++)
    {
        partial.init();
#pragma unroll
        for (SizeType32 i = tid; i < size; i += BLOCK_SIZE_)
        {
            partial.insert((float) sVal[i], i);
        }

        TopK_2<float> total = BlockReduce(tempStorage).Reduce(partial, reduce_topk_op_2<float>);

        if (tid == 0)
        {
            sId[ite] = total.p;
            sLogits[ite] = total.u;
            sVal[total.p] = -MAX_T_VAL;
            
            // Track max logit for numerical stability
            if (total.u > sMaxLogit)
            {
                sMaxLogit = total.u;
            }
        }
        __syncthreads();
    }

    // Phase 2: Compute probabilities only for selected top-K tokens
    if (tid == 0)
    {
        // Convert selected logits to probabilities
        for (SizeType32 i = 0; i < k; i++)
        {
            if (!logitHasProbs)
            {
                // Apply softmax only to selected tokens: exp(logit - max_logit)
                sProbs[i] = __expf(sLogits[i] - sMaxLogit);
            }
            else
            {
                // Logits are already probabilities
                sProbs[i] = sLogits[i];
            }
            sSum += sProbs[i];
        }
        
        // Normalize probabilities
        for (SizeType32 i = 0; i < k; i++)
        {
            sProbs[i] /= sSum;
        }
    }
    __syncthreads();

    // Phase 3: Sampling and output generation
    if (tid == 0)
    {
        auto randNum = (returnAllSelectedTokens || curandState == nullptr)
            ? static_cast<float>(probThreshold * sSum)
            : static_cast<float>(curand_uniform(curandState + batchSlot) * probThreshold * sSum);
        auto randNum2 = sampleTokenInSelected
            ? static_cast<float>(curand_uniform(curandState + batchSlot) * probThreshold * sSum)
            : 0.0f;
            
        auto* outputIdsRequestPtr = idsPtrs == nullptr ? ids + batchSlot * maxSeqLen : idsPtrs[batchSlot];
        
        float cumProb = 0.0f;
        for (SizeType32 ki = 0; ki < k; ki++)
        {
            cumProb += sProbs[ki];
            randNum -= sProbs[ki] * sSum; // Scale back for comparison
            if (sampleTokenInSelected)
            {
                randNum2 -= sProbs[ki] * sSum;
            }
            
            if (randNum <= 0.0f || ki == k - 1 || returnAllSelectedTokens)
            {
                auto idx = sId[ki];
                auto outputId = idx != -1
                    ? topKTmpIdBuf[(batchIdx * maxTokensPerStep + tokenIdx) * stride + idx] % vocabSize
                    : vocabSize - 1;
                outputId = outputId == -1 ? vocabSize - 1 : outputId;
                
                auto const curSeqLen = sequenceLengths == nullptr ? 0 : sequenceLengths[batchSlot];
                auto const outIdx = returnAllSelectedTokens ? tokenIdx * maxTopK + ki : curSeqLen + tokenIdx;
                outputIdsRequestPtr[outIdx] = outputId;

                if (returnAllSelectedTokens)
                {
                    // Directly use computed probability for output log probs
                    if (outputLogProbs != nullptr)
                    {
                        auto logProb = logf(sProbs[ki]);
                        // Probability is already normalized, so we can use it directly
                        // or apply additional normalization if needed
                        auto const finalLogProb = normalizeLogProbs ? logProb : logProb + logf(sSum);
                        outputLogProbs[batchSlot * maxTopK + ki] = finalLogProb;
                    }
                }
                else
                {
                    if (cumLogProbs != nullptr || outputLogProbs != nullptr)
                    {
                        auto logProb = logf(sProbs[ki]);
                        if (cumLogProbs != nullptr)
                        {
                            cumLogProbs[batchSlot] += normalizeLogProbs ? logProb : logProb + logf(sSum);
                        }
                        if (outputLogProbs != nullptr)
                        {
                            auto const finalLogProb = normalizeLogProbs ? logProb : logProb + logf(sSum);
                            outputLogProbs[curSeqLen * maxBatchSize + batchSlot] = finalLogProb;
                        }
                    }
                    break;
                }

                if (sampleTokenInSelected && randNum2 <= 0.0f)
                {
                    randNum2 = MAX_T_VAL;
                    outputIdCurrentStep[batchSlot] = outputId;
                }

                if (returnAllSelectedTokens && randNum <= 0.0f && strictTopPBoundary)
                {
                    if (ki < k - 1)
                    {
                        outputIdsRequestPtr[outIdx + 1] = -1;
                    }
                    break;
                }
            }
        }
        
        // Handle sequence completion
        if (maxTokensPerStep == 1 && !returnAllSelectedTokens && sequenceLengths != nullptr 
            && finishedOutput != nullptr && endIds != nullptr)
        {
            auto const seqLen = sequenceLengths[batchSlot];
            if (outputIdsRequestPtr[seqLen] == endIds[batchSlot])
            {
                finishedOutput[batchSlot].setFinishedEOS();
            }
            else
            {
                sequenceLengths[batchSlot] += 1;
            }
        }
    }
}

#define CASE_K(K_MAX, BLOCK_SIZE_1_, BLOCK_SIZE_2_, BLOCKS_PER_BEAM_)                                                  \
    do                                                                                                                 \
    {                                                                                                                  \
        {                                                                                                              \
            dim3 grid(params.batchSize* BLOCKS_PER_BEAM_, params.maxTokensPerStep);                                    \
            dim3 block(BLOCK_SIZE_1_);                                                                                 \
            topKStage1<T, BLOCK_SIZE_1_, BLOCKS_PER_BEAM_><<<grid, block, 0, stream>>>(params.logProbs,                \
                params.logProbsPtrs, tempLogProbs, topKTmpIdBuf, topKTmpValBuf, params.finishedInput, params.maxTopK,  \
                params.topKs, params.vocabSizePadded, params.endIds, params.skipDecode, params.batchSlots,             \
                params.tokensPerStep, params.maxTokensPerStep);                                                        \
        }                                                                                                              \
        {                                                                                                              \
            dim3 grid(params.batchSize, params.maxTokensPerStep);                                                      \
            dim3 block(BLOCK_SIZE_2_);                                                                                 \
            /* Increased shared memory: K_MAX indices + K_MAX logits + K_MAX probabilities */                         \
            size_t sharedMem = K_MAX * (sizeof(SizeType32) + 2 * sizeof(float));                                      \
            topKStage2Sampling<T, BLOCK_SIZE_2_, BLOCKS_PER_BEAM_>                                            \
                <<<grid, block, sharedMem, stream>>>(topKTmpIdBuf,                                                     \
                    topKTmpValBuf, params.outputIdsPtrs, params.outputIds, params.sequenceLengths,                     \
                    params.finishedInput, params.finishedOutput, params.cumLogProbs, params.outputLogProbs,            \
                    params.maxTopK, params.topKs, params.maxTopP, params.topPs, params.curandState, params.endIds,     \
                    params.vocabSizePadded, params.skipDecode, params.batchSlots, params.maxBatchSize,                 \
                    params.normalizeLogProbs, params.logitsHasProbs, params.tokensPerStep, params.maxTokensPerStep,    \
                    params.maxSeqLen, params.returnAllSelectedTokens, params.strictTopPBoundary,                       \
                    params.returnAllSelectedTokensPerSlot, params.outputIdCurrentStep,                                 \
                    params.skipOutputIdCurrentStep);                                                                   \
        }                                                                                                              \
    } while (0)

template <typename T>
void invokeBatchTopKSampling(TopKSamplingKernelParams<T> const& params, cudaStream_t stream)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    params.checkParams();

    // Not allow an ambiguous inputs topP and topPs.
    auto const workspaceSizes
        = getTopKWorkspaceSizes<T>(params.batchSize, params.maxTokensPerStep, params.maxTopK, params.vocabSizePadded);

    if (params.maxTopK == 0)
    {
        return;
    }

    std::vector<void*> alignedPointers;
    calcAlignedPointers(alignedPointers, params.workspace, workspaceSizes);

    auto tempLogProbs = static_cast<T*>(alignedPointers[0]);
    auto topKTmpIdBuf = static_cast<SizeType32*>(alignedPointers[1]);
    auto topKTmpValBuf = static_cast<T*>(alignedPointers[2]);

    SizeType32 logMaxTopK{0};
    SizeType32 recursor{params.maxTopK - 1};
    while (recursor >>= 1)
    {
        ++logMaxTopK;
    }

    switch (logMaxTopK)
    {
    case 0:
    case 1:
    case 2:
    case 3: // 0 < maxTopK <= 16
        CASE_K(16, 128, 128, 8);
        break;
    case 4: // 16 < maxTopK <= 32
        CASE_K(32, 256, 128, 8);
        break;
    case 5: // 32 < maxTopK <= 64
        CASE_K(64, 256, 256, 8);
        break;
    case 6:
    case 7:
    case 8:
    case 9: // 64 < maxTopK <= 1024
        CASE_K(1024, 256, 256, 8);
        break;
    default: TLLM_CHECK_WITH_INFO(false, "TopK kernel supports 1 <= k <= 1024 but got k=%d", params.maxTopK);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

#undef CASE_K

template void invokeBatchTopKSampling(TopKSamplingKernelParams<float> const& params, cudaStream_t stream);

template void invokeBatchTopKSampling(TopKSamplingKernelParams<half> const& params, cudaStream_t stream);

__global__ void setupTopKRuntimeArgs(SizeType32 batchSize, ScatterDecodingParamEntry<SizeType32> topK,
    ScatterDecodingParamEntry<float> topP, SizeType32 const* batchSlots, bool* skipDecode)
{
    auto const index = static_cast<SizeType32>(blockIdx.x * blockDim.x + threadIdx.x);
    for (auto bi = index; bi < batchSize; bi += static_cast<SizeType32>(gridDim.x * blockDim.x))
    {
        setupTopKTopPRuntimeArgOne(bi, topK, topP, batchSlots, skipDecode, nullptr, nullptr);
    }
}

void invokeSetupTopKRuntimeArgs(SizeType32 batchSize, ScatterDecodingParamEntry<SizeType32> topK,
    ScatterDecodingParamEntry<float> topP, bool* skipDecodePtr, SizeType32 const* batchSlotsPtr, bool onDevice,
    cudaStream_t stream)
{
    if (onDevice)
    {
        dim3 block(std::min(static_cast<uint32_t>(batchSize), 256u));
        dim3 grid(divUp(static_cast<uint32_t>(batchSize), block.x));
        // support topK up to TOP_K_MAX.
        setupTopKRuntimeArgs<<<grid, block, 0, stream>>>(batchSize, topK, topP, batchSlotsPtr, skipDecodePtr);
    }
    else
    {
        for (int bi = 0; bi < batchSize; ++bi)
        {
            setupTopKTopPRuntimeArgOne(bi, topK, topP, batchSlotsPtr, skipDecodePtr, nullptr, nullptr);
        }
    }
}

__global__ void setupTopKTopPRuntimeArgs(SizeType32 batchSize, ScatterDecodingParamEntry<SizeType32> topK,
    ScatterDecodingParamEntry<float> topP, SizeType32 const* batchSlots, bool* skipDecodeTopK, bool* skipDecodeTopP)
{
    auto const index = static_cast<SizeType32>(blockIdx.x * blockDim.x + threadIdx.x);
    for (auto bi = index; bi < batchSize; bi += static_cast<SizeType32>(gridDim.x * blockDim.x))
    {
        setupTopKTopPRuntimeArgOne(bi, topK, topP, batchSlots, skipDecodeTopK, skipDecodeTopP, nullptr);
    }
}

void invokeSetupTopKTopPRuntimeArgs(SizeType32 batchSize, ScatterDecodingParamEntry<SizeType32> topK,
    ScatterDecodingParamEntry<float> topP, bool* skipDecodeTopKPtr, bool* skipDecodeTopPPtr,
    SizeType32 const* batchSlotsPtr, bool onDevice, cudaStream_t stream)
{
    if (onDevice)
    {
        dim3 block(std::min(static_cast<uint32_t>(batchSize), 256u));
        dim3 grid(divUp(static_cast<uint32_t>(batchSize), block.x));
        // support topK up to TOP_K_MAX.
        setupTopKTopPRuntimeArgs<<<grid, block, 0, stream>>>(
            batchSize, topK, topP, batchSlotsPtr, skipDecodeTopKPtr, skipDecodeTopPPtr);
    }
    else
    {
        for (int bi = 0; bi < batchSize; ++bi)
        {
            setupTopKTopPRuntimeArgOne(bi, topK, topP, batchSlotsPtr, skipDecodeTopKPtr, skipDecodeTopPPtr, nullptr);
        }
    }
}

} // namespace tensorrt_llm::kernels
