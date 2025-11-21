#include "GPUTrainer.hpp"
#include <stdexcept>
#include <iostream>
#include <vector>
#include <cstring>
#include <string>

// ------------ Прототипы функций, реализованных в GPUTrainer.cu ------------

// Проверить доступность CUDA-устройства
bool cudaNNIsAvailable();

// Освободить все GPU-ресурсы (вызывается из деструктора)
void cudaNNFree();

// Инициализация сети на GPU
// layersSize: массив размеров слоёв (длина L)
// shapes: массив WeightShape размером L-1
// totalWeights / totalBiases: суммарное количество весов / смещений
// weightsFlat / biasesFlat: плоский массив всех весов / смещений
bool cudaNNInit(
    const uint32_t* layersSize,
    uint32_t L,
    const WeightShape* shapes,
    uint32_t numLayers,
    uint32_t totalWeights,
    uint32_t totalBiases,
    const float* weightsFlat,
    const float* biasesFlat
);

// Одна эпоха обучения (per-sample SGD)
// inputs:  N * nIn
// targets: N * nOut (one-hot)
// outMeanLoss: средний лосс по эпохе
bool cudaNNTrainEpoch(
    const float* inputs,
    const float* targets,
    uint32_t nSamples,
    float eta,
    float* outMeanLoss
);

// Скачивание всех весов / смещений из GPU в плоские массивы
bool cudaNNDownload(
    float* outWeightsFlat,
    float* outBiasesFlat
);

// -------------------------------------------------------------------

GPUTrainer& GPUTrainer::instance() {
    static GPUTrainer inst;
    return inst;
}

GPUTrainer::GPUTrainer() {
    ok = cudaNNIsAvailable();
    if (!ok) {
        std::cerr << "CUDA NN trainer: no compatible GPU device\n";
    }
}

GPUTrainer::~GPUTrainer() {
    if (ok) {
        cudaNNFree();
    }
}

void GPUTrainer::buildShapes() {
    uint32_t L = static_cast<uint32_t>(layersSize.size());
    if (L < 2) throw std::runtime_error("Network must have at least 2 layers");

    layerShapes.clear();
    layerShapes.resize(L - 1);

    totalWeights = 0;
    totalBiases = 0;

    for (uint32_t k = 0; k < L - 1; ++k) {
        uint32_t nIn = layersSize[k];
        uint32_t nOut = layersSize[k + 1];
        WeightShape ws;
        ws.nIn = nIn;
        ws.nOut = nOut;
        ws.wOffset = totalWeights;
        ws.bOffset = totalBiases;
        totalWeights += nIn * nOut;
        totalBiases += nOut;
        layerShapes[k] = ws;
    }
}

void GPUTrainer::initNetwork(
    const std::vector<uint32_t>& ls,
    const std::vector<std::vector<float>>& weightsCPU,
    const std::vector<std::vector<float>>& biasesCPU
) {
    if (!ok) throw std::runtime_error("GPUTrainer (CUDA): not available");

    layersSize = ls;
    uint32_t L = static_cast<uint32_t>(layersSize.size());
    if (L < 2) throw std::runtime_error("initNetwork: network must have at least 2 layers");

    if (weightsCPU.size() != L - 1 || biasesCPU.size() != L - 1)
        throw std::runtime_error("initNetwork: size mismatch (weights/biases)");

    buildShapes();

    // Плоские массивы весов/смещений
    std::vector<float> allW(totalWeights);
    std::vector<float> allB(totalBiases);

    for (uint32_t k = 0; k < layerShapes.size(); ++k) {
        const auto& ws = layerShapes[k];
        const auto& w = weightsCPU[k];
        const auto& b = biasesCPU[k];

        if (w.size() != static_cast<size_t>(ws.nIn) * ws.nOut)
            throw std::runtime_error("initNetwork: weights size mismatch in layer " + std::to_string(k));
        if (b.size() != ws.nOut)
            throw std::runtime_error("initNetwork: biases size mismatch in layer " + std::to_string(k));

        std::memcpy(allW.data() + ws.wOffset,
            w.data(),
            sizeof(float) * ws.nIn * ws.nOut);
        std::memcpy(allB.data() + ws.bOffset,
            b.data(),
            sizeof(float) * ws.nOut);
    }

    bool okInit = cudaNNInit(
        layersSize.data(),
        L,
        layerShapes.data(),
        static_cast<uint32_t>(layerShapes.size()),
        totalWeights,
        totalBiases,
        allW.data(),
        allB.data()
    );
    if (!okInit) {
        ok = false;
        throw std::runtime_error("cudaNNInit failed");
    }
}

void GPUTrainer::trainEpoch(
    const std::vector<std::pair<std::vector<float>, std::vector<float>>>& data,
    float eta
) {
    if (!ok) throw std::runtime_error("GPUTrainer (CUDA): not available");
    if (data.empty()) return;

    uint32_t N = static_cast<uint32_t>(data.size());
    uint32_t nIn = layersSize.front();
    uint32_t nOut = layersSize.back();

    // Упаковываем входы и таргеты в плоские массивы
    std::vector<float> flatInputs;
    std::vector<float> flatTargets;
    flatInputs.resize(static_cast<size_t>(N) * nIn);
    flatTargets.resize(static_cast<size_t>(N) * nOut);

    for (uint32_t i = 0; i < N; ++i) {
        const auto& in = data[i].first;
        const auto& tgt = data[i].second;
        if (in.size() != nIn || tgt.size() != nOut)
            throw std::runtime_error("trainEpoch: data size mismatch");

        std::memcpy(flatInputs.data() + static_cast<size_t>(i) * nIn,
            in.data(),
            sizeof(float) * nIn);
        std::memcpy(flatTargets.data() + static_cast<size_t>(i) * nOut,
            tgt.data(),
            sizeof(float) * nOut);
    }

    float meanLoss = 0.0f;
    bool okTrain = cudaNNTrainEpoch(
        flatInputs.data(),
        flatTargets.data(),
        N,
        eta,
        &meanLoss
    );
    if (!okTrain)
        throw std::runtime_error("cudaNNTrainEpoch failed");

    std::cout << "epoch mean loss = " << meanLoss << "\n";
}

void GPUTrainer::downloadNetwork(
    std::vector<std::vector<float>>& weightsCPU,
    std::vector<std::vector<float>>& biasesCPU
) {
    if (!ok) throw std::runtime_error("GPUTrainer (CUDA): not available");

    std::vector<float> allW(totalWeights);
    std::vector<float> allB(totalBiases);

    bool okDown = cudaNNDownload(allW.data(), allB.data());
    if (!okDown)
        throw std::runtime_error("cudaNNDownload failed");

    uint32_t Lw = static_cast<uint32_t>(layerShapes.size());
    weightsCPU.resize(Lw);
    biasesCPU.resize(Lw);

    for (uint32_t k = 0; k < Lw; ++k) {
        const auto& ws = layerShapes[k];
        weightsCPU[k].resize(static_cast<size_t>(ws.nIn) * ws.nOut);
        biasesCPU[k].resize(ws.nOut);

        std::memcpy(weightsCPU[k].data(),
            allW.data() + ws.wOffset,
            sizeof(float) * ws.nIn * ws.nOut);
        std::memcpy(biasesCPU[k].data(),
            allB.data() + ws.bOffset,
            sizeof(float) * ws.nOut);
    }
}
