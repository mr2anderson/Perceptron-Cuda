#pragma once

#include <vector>
#include <cstdint>

class Tensor2D {
public:
    Tensor2D();
    Tensor2D(uint32_t a, uint32_t b);

    float& operator()(uint32_t x, uint32_t y);
    float  operator()(uint32_t x, uint32_t y) const;

    float& operator()(uint32_t i);
    const float& operator()(uint32_t i) const;

    uint32_t getA() const;
    uint32_t getB() const;
    uint32_t getSize() const;

private:
    uint32_t a, b;
    std::vector<float> data;
};
