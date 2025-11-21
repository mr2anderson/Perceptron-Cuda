#pragma once

#include <vector>
#include <cstdint>
#include <utility>

struct WeightShape {
    uint32_t nIn;
    uint32_t nOut;
    // смещение в глобальном массиве весов / градиентов (в float'ах)
    uint32_t wOffset;
    uint32_t bOffset;
};

class GPUTrainer {
public:
    static GPUTrainer& instance();

    bool isAvailable() const { return ok; }

    // »нициализировать GPU-копию сети
    // layersSize: размеры слоЄв (как в Perceptron::layersSize)
    // weightsCPU[k]: матрица k-го сло€ в виде [i * nOut + j]
    // biasesCPU[k]: вектор смещений дл€ сло€ k
    void initNetwork(
        const std::vector<uint32_t>& layersSize,
        const std::vector<std::vector<float>>& weightsCPU,
        const std::vector<std::vector<float>>& biasesCPU
    );

    // ќдна эпоха обучени€ по всему датасету (full-batch SGD)
    // data: (input, target), target Ч one-hot
    void trainEpoch(
        const std::vector<std::pair<std::vector<float>, std::vector<float>>>& data,
        float eta
    );

    // —качать веса и смещени€ из GPU обратно в CPU
    void downloadNetwork(
        std::vector<std::vector<float>>& weightsCPU,
        std::vector<std::vector<float>>& biasesCPU
    );

    const std::vector<uint32_t>& getLayersSize() const { return layersSize; }

private:
    GPUTrainer();
    ~GPUTrainer();
    GPUTrainer(const GPUTrainer&) = delete;
    GPUTrainer& operator=(const GPUTrainer&) = delete;

    void buildShapes(); // заполн€ет layerShapes / totalWeights / totalBiases

private:
    bool ok = false;

    // ќписание сети (на CPU)
    std::vector<uint32_t> layersSize;     // размеры слоЄв
    std::vector<WeightShape> layerShapes; // размеры и смещени€ слоЄв

    uint32_t totalWeights = 0;
    uint32_t totalBiases  = 0;
};
