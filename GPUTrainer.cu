#include "GPUTrainer.hpp"

#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <cstdint>

#define CUDA_CHECK(expr)                                                      \
    do {                                                                      \
        cudaError_t err__ = (expr);                                           \
        if (err__ != cudaSuccess) {                                           \
            std::cerr << "CUDA error: " << cudaGetErrorString(err__)          \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";       \
            return false;                                                     \
        }                                                                     \
    } while (0)

#define CUDA_CHECK_VOID(expr)                                                 \
    do {                                                                      \
        cudaError_t err__ = (expr);                                           \
        if (err__ != cudaSuccess) {                                           \
            std::cerr << "CUDA error: " << cudaGetErrorString(err__)          \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";       \
        }                                                                     \
    } while (0)


// ---------------- Глобальное состояние на GPU ----------------

static bool gAvailable = false;
static bool gInitialized = false;

static std::vector<uint32_t>    gLayersSize;   // размеры слоёв, длина L
static std::vector<WeightShape> gShapes;       // L-1 элементов

static uint32_t gL = 0;
static uint32_t gTotalWeights = 0;
static uint32_t gTotalBiases = 0;

// активации / z / delta для ОДНОГО образца
static std::vector<uint32_t> gNeuronOffset;    // смещения слоёв в общих массивах a/z/delta
static uint32_t gTotalNeurons = 0;

// GPU-память
static float* d_weights = nullptr;
static float* d_biases = nullptr;

static float* d_a = nullptr;   // activations
static float* d_z = nullptr;   // pre-activations
static float* d_delta = nullptr;   // deltas

static float* d_target = nullptr;   // целевой вектор (nOut)

// ---------------- CUDA kernels ----------------

// forward одного слоя (один образец)
// aPrev: [nIn]
// W: [nIn * nOut] (строка i, столбец j: W[i * nOut + j])
// B: [nOut]
// zNext: [nOut], aNext: [nOut]
__global__
void forwardLayerKernel(const float* aPrev,
    const float* W,
    const float* B,
    float* zNext,
    float* aNext,
    int nIn,
    int nOut,
    int applyRelu)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= nOut) return;

    float s = B[j];
    for (int i = 0; i < nIn; ++i) {
        float w = W[i * nOut + j];
        float v = aPrev[i];
        s += w * v;
    }

    zNext[j] = s;
    if (applyRelu) {
        aNext[j] = (s > 0.0f ? s : 0.0f);
    }
    else {
        aNext[j] = s;
    }
}

// softmax по последнему слою (один образец)
__global__
void softmaxKernel(float* aLast,
    int nOut)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    // максимум
    float m = aLast[0];
    for (int j = 1; j < nOut; ++j) {
        if (aLast[j] > m) m = aLast[j];
    }
    // экспоненты и сумма
    float s = 0.0f;
    for (int j = 0; j < nOut; ++j) {
        float v = expf(aLast[j] - m);
        aLast[j] = v;
        s += v;
    }
    if (s == 0.0f) {
        float u = 1.0f / float(nOut);
        for (int j = 0; j < nOut; ++j)
            aLast[j] = u;
        return;
    }
    float inv = 1.0f / s;
    for (int j = 0; j < nOut; ++j)
        aLast[j] *= inv;
}

// delta_L[j] = a_L[j] - target[j]
__global__
void lastDeltaKernel(const float* aLast,
    const float* target,
    float* deltaLast,
    int nOut)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= nOut) return;
    deltaLast[j] = aLast[j] - target[j];
}

// deltaPrev[i] = (Σ_j deltaNext[j] * W[i,j]) * relu'(zPrev[i])
__global__
void backwardHiddenKernel(const float* zPrev,
    const float* W,
    const float* deltaNext,
    float* deltaPrev,
    int nIn,
    int nOut)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nIn) return;

    float s = 0.0f;
    for (int j = 0; j < nOut; ++j) {
        float d = deltaNext[j];
        float w = W[i * nOut + j];
        s += d * w;
    }

    float z = zPrev[i];
    float rp = (z > 0.0f) ? 1.0f : 0.0f;
    deltaPrev[i] = s * rp;
}

// обновление весов: W[i,j] -= eta * aPrev[i] * deltaNext[j]
__global__
void updateWeightsKernel(float* W,
    const float* aPrev,
    const float* deltaNext,
    int nIn,
    int nOut,
    float eta)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nIn || j >= nOut) return;

    float a = aPrev[i];
    float d = deltaNext[j];
    int idx = i * nOut + j;
    W[idx] -= eta * a * d;
}

// обновление смещений: B[j] -= eta * deltaNext[j]
__global__
void updateBiasKernel(float* B,
    const float* deltaNext,
    int nOut,
    float eta)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= nOut) return;
    B[j] -= eta * deltaNext[j];
}

// ---------------- Host API ----------------

bool cudaNNIsAvailable() {
    int devCount = 0;
    cudaError_t err = cudaGetDeviceCount(&devCount);
    if (err != cudaSuccess || devCount == 0) {
        gAvailable = false;
        return false;
    }
    gAvailable = true;
    return true;
}

void cudaNNFree() {
    if (!gInitialized) return;

    CUDA_CHECK_VOID(cudaFree(d_weights));
    CUDA_CHECK_VOID(cudaFree(d_biases));
    CUDA_CHECK_VOID(cudaFree(d_a));
    CUDA_CHECK_VOID(cudaFree(d_z));
    CUDA_CHECK_VOID(cudaFree(d_delta));
    CUDA_CHECK_VOID(cudaFree(d_target));

    d_weights = d_biases = nullptr;
    d_a = d_z = d_delta = nullptr;
    d_target = nullptr;

    gInitialized = false;
}

bool cudaNNInit(
    const uint32_t* layersSize,
    uint32_t L,
    const WeightShape* shapes,
    uint32_t numLayers,
    uint32_t totalWeights,
    uint32_t totalBiases,
    const float* weightsFlat,
    const float* biasesFlat
) {
    if (!gAvailable && !cudaNNIsAvailable())
        return false;

    cudaNNFree(); // на всякий случай

    gL = L;
    gLayersSize.assign(layersSize, layersSize + L);
    gShapes.assign(shapes, shapes + numLayers);
    gTotalWeights = totalWeights;
    gTotalBiases = totalBiases;

    // считаем оффсеты нейронов по слоям
    gNeuronOffset.clear();
    gNeuronOffset.resize(L);
    gTotalNeurons = 0;
    for (uint32_t k = 0; k < L; ++k) {
        gNeuronOffset[k] = gTotalNeurons;
        gTotalNeurons += gLayersSize[k];
    }

    // выделяем память под веса/смещения
    CUDA_CHECK(cudaMalloc(&d_weights, gTotalWeights * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_biases, gTotalBiases * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_weights, weightsFlat,
        gTotalWeights * sizeof(float),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_biases, biasesFlat,
        gTotalBiases * sizeof(float),
        cudaMemcpyHostToDevice));

    // выделяем память под a/z/delta и target (для одного образца)
    CUDA_CHECK(cudaMalloc(&d_a, gTotalNeurons * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_z, gTotalNeurons * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_delta, gTotalNeurons * sizeof(float)));

    uint32_t nOut = gLayersSize.back();
    CUDA_CHECK(cudaMalloc(&d_target, nOut * sizeof(float)));

    // обнулим
    CUDA_CHECK(cudaMemset(d_a, 0, gTotalNeurons * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_z, 0, gTotalNeurons * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_delta, 0, gTotalNeurons * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_target, 0, nOut * sizeof(float)));

    gInitialized = true;
    return true;
}

bool cudaNNTrainEpoch(
    const float* inputs,
    const float* targets,
    uint32_t nSamples,
    float eta,
    float* outMeanLoss
) {
    if (!gInitialized) {
        std::cerr << "cudaNNTrainEpoch: network not initialized\n";
        return false;
    }
    if (nSamples == 0) {
        *outMeanLoss = 0.0f;
        return true;
    }

    uint32_t nIn = gLayersSize.front();
    uint32_t nOut = gLayersSize.back();

    double lossSum = 0.0;

    for (uint32_t s = 0; s < nSamples; ++s) {
        const float* inHost = inputs + static_cast<size_t>(s) * nIn;
        const float* tgtHost = targets + static_cast<size_t>(s) * nOut;

        if ((s & 31u) == 0u || s + 1 == nSamples) {
            float pct = (float)(s + 1) * 100.0f / (float)nSamples;
            printf("\repoch: %.1f%%", pct, s + 1, nSamples);
            fflush(stdout);
        }

        // input -> a[0]
        CUDA_CHECK(cudaMemcpy(
            d_a + gNeuronOffset[0],
            inHost,
            nIn * sizeof(float),
            cudaMemcpyHostToDevice
        ));

        // target -> d_target
        CUDA_CHECK(cudaMemcpy(
            d_target,
            tgtHost,
            nOut * sizeof(float),
            cudaMemcpyHostToDevice
        ));

        // --------- forward ---------
        {
            for (uint32_t k = 0; k < gL - 1; ++k) {
                const auto& ws = gShapes[k];
                int nInL = static_cast<int>(ws.nIn);
                int nOutL = static_cast<int>(ws.nOut);

                const float* aPrev = d_a + gNeuronOffset[k];
                float* aNext = d_a + gNeuronOffset[k + 1];
                float* zNext = d_z + gNeuronOffset[k + 1];

                const float* Wk = d_weights + ws.wOffset;
                const float* Bk = d_biases + ws.bOffset;

                int applyRelu = (k < gL - 2) ? 1 : 0; // все кроме последнего

                int threads = 256;
                int blocks = (nOutL + threads - 1) / threads;

                forwardLayerKernel << <blocks, threads >> > (
                    aPrev, Wk, Bk,
                    zNext, aNext,
                    nInL, nOutL,
                    applyRelu
                    );
                if (cudaPeekAtLastError() != cudaSuccess) {
                    std::cerr << "FORWARD LAYER KERNEL FAIL\n";
                    return false;
                }
            }

            // softmax на последнем слое
            float* aLast = d_a + gNeuronOffset[gL - 1];
            softmaxKernel << <1, 1 >> > (aLast, static_cast<int>(nOut));
 
            if (cudaPeekAtLastError() != cudaSuccess) {
                std::cerr << "SOFTMAX KERNEL FAIL\n";
                return false;
            }

            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // --------- loss (на CPU) ---------
        {
            std::vector<float> outHost(nOut);
            CUDA_CHECK(cudaMemcpy(
                outHost.data(),
                d_a + gNeuronOffset[gL - 1],
                nOut * sizeof(float),
                cudaMemcpyDeviceToHost
            ));

            // находим класс-метку по target (one-hot)
            uint32_t label = 0;
            for (uint32_t j = 0; j < nOut; ++j) {
                if (std::fabs(tgtHost[j] - 1.0f) < 1e-6f) {
                    label = j;
                    break;
                }
            }
            float p = outHost[label];
            if (p < 1e-9f) p = 1e-9f;
            lossSum += -std::log(p);
        }

        // --------- backward ---------
        {
            const float* aLast = d_a + gNeuronOffset[gL - 1];
            float* deltaLast = d_delta + gNeuronOffset[gL - 1];

            // delta_L = a_L - target
            {
                int threads = 256;
                int blocks = (static_cast<int>(nOut) + threads - 1) / threads;
                lastDeltaKernel << <blocks, threads >> > (
                    aLast,
                    d_target,
                    deltaLast,
                    static_cast<int>(nOut)
                    );
                if (cudaPeekAtLastError() != cudaSuccess) {
                    std::cerr << "LAST DELTA KERNEL FAIL\n";
                    return false;
                }
            }

            // скрытые слои: k = L-2 .. 1
            for (int32_t k = static_cast<int32_t>(gL) - 2; k > 0; --k) {
                const auto& ws = gShapes[static_cast<uint32_t>(k)];
                int nInL = static_cast<int>(ws.nIn);
                int nOutL = static_cast<int>(ws.nOut);

                const float* zPrev = d_z + gNeuronOffset[static_cast<uint32_t>(k)];
                const float* deltaNext = d_delta + gNeuronOffset[static_cast<uint32_t>(k) + 1];
                float* deltaPrev = d_delta + gNeuronOffset[static_cast<uint32_t>(k)];

                int threads = 256;
                int blocks = (nInL + threads - 1) / threads;

                backwardHiddenKernel << <blocks, threads >> > (
                    zPrev,
                    d_weights + ws.wOffset,
                    deltaNext,
                    deltaPrev,
                    nInL,
                    nOutL
                    );
                if (cudaPeekAtLastError() != cudaSuccess) {
                    std::cerr << "BACKWARD HIDDEN KERNEL FAIL\n";
                    return false;
                }
            }

            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // --------- update (SGD per sample) ---------
        {
            uint32_t numLayers = static_cast<uint32_t>(gShapes.size());
            for (uint32_t k = 0; k < numLayers; ++k) {
                const auto& ws = gShapes[k];
                int nInL = static_cast<int>(ws.nIn);
                int nOutL = static_cast<int>(ws.nOut);

                float* Wk = d_weights + ws.wOffset;
                float* Bk = d_biases + ws.bOffset;
                const float* aPrev = d_a + gNeuronOffset[k];
                const float* deltaNext = d_delta + gNeuronOffset[k + 1];

                dim3 blockW(16, 16);
                dim3 gridW(
                    (nOutL + blockW.x - 1) / blockW.x,
                    (nInL + blockW.y - 1) / blockW.y
                );
                updateWeightsKernel << <gridW, blockW >> > (
                    Wk, aPrev, deltaNext,
                    nInL, nOutL,
                    eta
                    );
                if (cudaPeekAtLastError() != cudaSuccess) {
                    std::cerr << "UPDATE WEIGHTS KERNEL FAIL\n";
                    return false;
                }

                int threadsB = 256;
                int blocksB = (nOutL + threadsB - 1) / threadsB;
                updateBiasKernel << <blocksB, threadsB >> > (
                    Bk,
                    deltaNext,
                    nOutL,
                    eta
                    );
                if (cudaPeekAtLastError() != cudaSuccess) {
                    std::cerr << "UPDATE BIASES KERNEL FAIL\n";
                    return false;
                }
            }

            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    printf("\r%*s\r", 60, "");
    fflush(stdout);

    *outMeanLoss = static_cast<float>(lossSum / static_cast<double>(nSamples));
    return true;
}

bool cudaNNDownload(
    float* outWeightsFlat,
    float* outBiasesFlat
) {
    if (!gInitialized) {
        std::cerr << "cudaNNDownload: network not initialized\n";
        return false;
    }

    CUDA_CHECK(cudaMemcpy(
        outWeightsFlat,
        d_weights,
        gTotalWeights * sizeof(float),
        cudaMemcpyDeviceToHost
    ));
    CUDA_CHECK(cudaMemcpy(
        outBiasesFlat,
        d_biases,
        gTotalBiases * sizeof(float),
        cudaMemcpyDeviceToHost
    ));
    return true;
}
