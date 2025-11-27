#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <cstdint>
#include <utility>
#include <algorithm>
#include <random>
#include "Perceptron.hpp"


struct CifarItem {
    std::vector<float> x;
    std::vector<float> y;
};


static std::vector<CifarItem> loadCifar(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        std::cout << "Cannot open file: " << path << "\n";
        return {};
    }

    const uint32_t imageBytes = 3072;
    std::vector<CifarItem> out;
    out.reserve(10000);

    for (uint32_t i = 0; i < 10000; i++) {
        uint8_t label;
        f.read(reinterpret_cast<char*>(&label), 1);

        CifarItem it;
        it.x.resize(imageBytes);
        it.y.assign(10, 0.0f);
        it.y[label] = 1.0f;

        std::vector<uint8_t> buf(imageBytes);
        f.read(reinterpret_cast<char*>(buf.data()), imageBytes);

        for (uint32_t j = 0; j < imageBytes; j++)
            it.x[j] = float(buf[j]) / 255.0f;

        out.push_back(std::move(it));
    }

    return out;
}


float accuracy(const Perceptron& p, const std::vector<CifarItem>& test) {
    const uint32_t total = test.size();
    uint32_t ok = 0;
    const uint32_t step = 500;
    for (uint32_t i = 0; i < total; i++) {
        auto out = p.eval(test[i].x);
        uint32_t pred = 0;
        float mx = out[0];
        for (uint32_t j = 1; j < 10; j++) if (out[j] > mx) mx = out[j], pred = j;
        uint32_t real = 0;
        for (uint32_t j = 0; j < 10; j++) if (test[i].y[j] == 1.0f) real = j;
        if (pred == real) ok++;
        if (i % step == 0) {
            int perc = int(float(i) / float(total) * 100.0f);
            std::cout << "\r[" << perc << "%] " << i << "/" << total << std::flush;
        }
    }
    std::cout << "\r" << std::string(40, ' ') << "\r";
    return float(ok) / float(total);
}




int main() {
    auto b1 = loadCifar("data/data_batch_1.bin");
    auto b2 = loadCifar("data/data_batch_2.bin");
    auto b3 = loadCifar("data/data_batch_3.bin");
    auto b4 = loadCifar("data/data_batch_4.bin");
    auto b5 = loadCifar("data/data_batch_5.bin");
    auto test = loadCifar("data/test_batch.bin");

    std::vector<CifarItem> train;
    train.reserve(50000);

    train.insert(train.end(), b1.begin(), b1.end());
    train.insert(train.end(), b2.begin(), b2.end());
    train.insert(train.end(), b3.begin(), b3.end());
    train.insert(train.end(), b4.begin(), b4.end());
    train.insert(train.end(), b5.begin(), b5.end());

    std::cout << "Train size: " << train.size() << std::endl;
    std::cout << "Test size: " << test.size() << std::endl;

    std::vector<std::pair<std::vector<float>, std::vector<float>>> trainData;
    trainData.reserve(train.size());

    for (auto& i : train)
        trainData.push_back({ i.x, i.y });

    std::vector<uint32_t> pConf = {
        3072,
        4096,
        2048,
        1024,
        512,
        256,
        10
    };
    Perceptron p(pConf);
    
    p.show();

    const float eta = 0.001f;
    const uint32_t maxEpoch = -1;

    p.train(
        trainData,
        eta,
        maxEpoch,

        [&](uint32_t epochN, const Perceptron& th) {
            if (epochN % 5 == 0) {
                float acc = accuracy(th, test);
                std::cout << "epoch test accuracy = " << acc << std::endl;
                std::string n = "autosave-" + std::to_string(epochN) + ".model";
                th.save(n);
                std::cout << "saved as " << n << std::endl;
            }
            
        }
    );

    return 0;
}
