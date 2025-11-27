#include "Perceptron.hpp"
#include <random>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <random>
#include "GPUTrainer.hpp"


Perceptron::Perceptron() = default;

Perceptron::Perceptron(const std::vector<uint32_t>& ls) {
    layersSize = ls;
    weights.resize(layersSize.size() - 1);
    biases.resize(layersSize.size() - 1);
    for (uint32_t i = 0; i < weights.size(); i++) {
        weights[i] = randomWeights(layersSize[i], layersSize[i + 1]);
        biases[i] = randomBiases(layersSize[i + 1]);
    }
}

Perceptron::Perceptron(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open perceptron file");

    uint32_t L = readInt(f);
    layersSize.resize(L);
    weights.resize(L - 1);
    biases.resize(L - 1);

    for (uint32_t i = 0; i < L; i++) layersSize[i] = readInt(f);

    for (uint32_t k = 0; k < L - 1; k++) {
        uint32_t a = layersSize[k];
        uint32_t b = layersSize[k + 1];
        weights[k] = Tensor2D(a, b);
        for (uint32_t i = 0; i < a * b; i++) {
            weights[k](i) = readFloat(f);
        }
    }

    for (uint32_t k = 0; k < L - 1; k++) {
        biases[k].resize(layersSize[k + 1]);
        for (uint32_t j = 0; j < biases[k].size(); j++) {
            biases[k][j] = readFloat(f);
        }
    }
}

std::vector<float> Perceptron::eval(std::vector<float> x) const {
    if (x.size() != layersSize[0]) {
        throw std::runtime_error("eval: input size mismatch");
    }

    std::vector<float> z;
    for (uint32_t k = 1; k < layersSize.size(); k++) {
        z.resize(layersSize[k]);
        inputSignals(x, weights[k - 1], biases[k - 1], z);
        if (k + 1 == layersSize.size()) {
            softmax(z);
            x = z;
        }
        else {
            relu(z);
            x = z;
        }
    }
    return x;
}

void Perceptron::train(
    std::vector<std::pair<std::vector<float>, std::vector<float>>>& data,
    float eta,
    uint32_t maxEpoch,
    const std::function<void(uint32_t, const Perceptron&)>& onEpochEnd
) {
    if (data.empty()) return;

    GPUTrainer& gpu = GPUTrainer::instance();
    if (!gpu.isAvailable()) {
        throw std::runtime_error("GPUTrainer is not available");
    }

    uint32_t L = (uint32_t)layersSize.size();

    // подготовить плоские веса/смещения для GPUTrainer
    std::vector<std::vector<float>> wFlat(L - 1);
    std::vector<std::vector<float>> bFlat(L - 1);

    for (uint32_t k = 0; k < L - 1; ++k) {
        uint32_t nIn = layersSize[k];
        uint32_t nOut = layersSize[k + 1];
        wFlat[k].resize(nIn * nOut);
        for (uint32_t i = 0; i < nIn; ++i)
            for (uint32_t j = 0; j < nOut; ++j)
                wFlat[k][i * nOut + j] = weights[k](i, j);
        bFlat[k] = biases[k];
    }

    gpu.initNetwork(layersSize, wFlat, bFlat);

    std::random_device rd;
    std::mt19937 mt(rd());

    for (uint32_t epoch = 1; epoch <= maxEpoch; ++epoch) {
        std::shuffle(data.begin(), data.end(), mt);

        gpu.trainEpoch(data, eta);

        // забираем обновлённые веса и сдвиги
        gpu.downloadNetwork(wFlat, bFlat);
        for (uint32_t k = 0; k < L - 1; ++k) {
            uint32_t nIn = layersSize[k];
            uint32_t nOut = layersSize[k + 1];
            for (uint32_t i = 0; i < nIn; ++i)
                for (uint32_t j = 0; j < nOut; ++j)
                    weights[k](i, j) = wFlat[k][i * nOut + j];
            biases[k] = bFlat[k];
        }

        if (onEpochEnd) {
            onEpochEnd(epoch, *this);
        }
    }
}

void Perceptron::save(const std::string& path) const {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open perceptron file for save");

    writeInt(f, static_cast<uint32_t>(layersSize.size()));
    for (uint32_t v : layersSize) writeInt(f, v);

    for (uint32_t k = 0; k < weights.size(); k++)
        for (uint32_t i = 0; i < weights[k].getSize(); i++)
            writeFloat(f, weights[k](i));

    for (uint32_t k = 0; k < biases.size(); k++)
        for (float v : biases[k])
            writeFloat(f, v);
}

void Perceptron::show() const {
    std::cout << Separator;
    uint32_t neurons = 0;
    for (uint32_t v : layersSize) {
        std::cout << "Layer: " << v << " neurons\n";
        neurons += v;
    }
    std::cout << Separator;

    uint32_t params = 0;
    for (uint32_t k = 0; k < weights.size(); k++) {
        uint32_t p = weights[k].getSize();
        std::cout << "Weights " << k << ": " << p << "\n";
        params += p;
    }
    for (uint32_t k = 0; k < biases.size(); k++) {
        uint32_t p = static_cast<uint32_t>(biases[k].size());
        std::cout << "Biases  " << k << ": " << p << "\n";
        params += p;
    }
    std::cout << Separator;
    std::cout << "Total neurons: " << neurons << "\n";
    std::cout << "Total params:  " << params
        << " (" << (params * sizeof(float)) / 1024.0 / 1024.0 << " MB)\n";
    std::cout << Separator;
}

void Perceptron::inputSignals(
    const std::vector<float>& layer,
    const Tensor2D& w,
    const std::vector<float>& b,
    std::vector<float>& out
) {
    uint32_t nIn = w.getA();
    uint32_t nOut = w.getB();
    if (layer.size() != nIn || b.size() != nOut) {
        throw std::runtime_error("inputSignals: size mismatch");
    }
    if (out.size() != nOut) out.resize(nOut);

    for (uint32_t j = 0; j < nOut; j++) out[j] = b[j];
    for (uint32_t i = 0; i < nIn; i++) {
        for (uint32_t j = 0; j < nOut; j++) {
            out[j] += layer[i] * w(i, j);
        }
    }
}

void Perceptron::relu(std::vector<float>& x) {
    for (float& v : x) {
        if (v < 0.f) v = 0.f;
    }
}

void Perceptron::softmax(std::vector<float>& x) {
    if (x.empty()) return;
    float m = *std::max_element(x.begin(), x.end());
    float s = 0.f;
    for (float& v : x) {
        v = std::exp(v - m);
        s += v;
    }
    if (s == 0.f) {
        float v = 1.f / static_cast<float>(x.size());
        for (float& t : x) t = v;
    }
    else {
        float inv = 1.f / s;
        for (float& v : x) v *= inv;
    }
}

Tensor2D Perceptron::randomWeights(uint32_t a, uint32_t b) {
    std::mt19937 rng(std::random_device{}());
    float limit = std::sqrt(6.f / static_cast<float>(a + b));
    std::uniform_real_distribution<float> dist(-limit, limit);
    Tensor2D t(a, b);
    for (uint32_t i = 0; i < t.getSize(); i++) {
        t(i) = dist(rng);
    }
    return t;
}

std::vector<float> Perceptron::randomBiases(uint32_t n) {
    return std::vector<float>(n, 0.f);
}

void Perceptron::writeInt(std::ofstream& os, uint32_t v) {
    os.write(reinterpret_cast<char*>(&v), sizeof(v));
}

uint32_t Perceptron::readInt(std::ifstream& is) {
    uint32_t x;
    is.read(reinterpret_cast<char*>(&x), sizeof(x));
    return x;
}

void Perceptron::writeFloat(std::ofstream& os, float v) {
    os.write(reinterpret_cast<char*>(&v), sizeof(v));
}

float Perceptron::readFloat(std::ifstream& is) {
    float v;
    is.read(reinterpret_cast<char*>(&v), sizeof(v));
    return v;
}
