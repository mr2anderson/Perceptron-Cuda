#pragma once

#include <vector>
#include <string>
#include <string_view>
#include <cstdint>
#include <utility>
#include <fstream>
#include "Tensor2D.hpp"

class Perceptron {
public:
    Perceptron();
    Perceptron(const std::vector<uint32_t>& layersSize);
    Perceptron(const std::string& path);

    std::vector<float> eval(std::vector<float> l1) const;

    void train(
        const std::vector<std::pair<std::vector<float>, std::vector<float>>>& data,
        float eta,
        uint32_t maxEpoch
    );

    void save(const std::string& path) const;
    void show() const;

private:
    static float reluDerivative(float x);
    static void inputSignals(
        const std::vector<float>& layer,
        const Tensor2D& weights,
        const std::vector<float>& biases,
        std::vector<float>& out
    );
    static void relu(std::vector<float>& layer);
    static void softmax(std::vector<float>& layer);
    static void lastLayerError(
        const std::vector<float>& layer,
        const std::vector<float>& perfect,
        std::vector<float>& error
    );
    static void hiddenLayerError(
        const std::vector<float>& layer,
        const Tensor2D& weights,
        const std::vector<float>& nextError,
        std::vector<float>& outError
    );
    static float Loss(const std::vector<float>& pred, const std::vector<float>& target);
    static void updateWeights(
        Tensor2D& weights,
        const std::vector<float>& error,
        const std::vector<float>& layer,
        float eta
    );
    static void updateBiases(
        std::vector<float>& biases,
        const std::vector<float>& error,
        float eta
    );
    static Tensor2D randomWeights(uint32_t left, uint32_t right);
    static std::vector<float> randomBiases(uint32_t sz);
    static void writeInt(std::ofstream& os, uint32_t v);
    static uint32_t readInt(std::ifstream& is);
    static void writeFloat(std::ofstream& os, float v);
    static float readFloat(std::ifstream& is);

    std::vector<uint32_t> layersSize;
    std::vector<Tensor2D> weights;
    std::vector<std::vector<float>> biases;

    static constexpr std::string_view Separator =
        "################################################################################\n";
};
