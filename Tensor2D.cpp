#include "Tensor2D.hpp"

Tensor2D::Tensor2D() = default;

Tensor2D::Tensor2D(uint32_t a_, uint32_t b_)
    : a(a_), b(b_), data(static_cast<size_t>(a_)* b_) {
}

float& Tensor2D::operator()(uint32_t x, uint32_t y) {
    return data[static_cast<size_t>(x) * b + y];
}

float Tensor2D::operator()(uint32_t x, uint32_t y) const {
    return data[static_cast<size_t>(x) * b + y];
}

float& Tensor2D::operator()(uint32_t i) {
    return data[i];
}

const float& Tensor2D::operator()(uint32_t i) const {
    return data[i];
}

uint32_t Tensor2D::getA() const { return a; }
uint32_t Tensor2D::getB() const { return b; }
uint32_t Tensor2D::getSize() const { return static_cast<uint32_t>(data.size()); }
