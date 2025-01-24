#pragma once

#include "onnx_model.hpp"
#include <string>
#include <vector>
#include <memory>

class CubicSpline {
public:
    CubicSpline(
        const std::vector<float>& x_knots,
        const std::vector<float>& a_coef,
        const std::vector<float>& b_coef,
        const std::vector<float>& c_coef,
        const std::vector<float>& d_coef,
        std::shared_ptr<ONNXModel> model
    );

    // Evaluate spline at given points
    std::vector<float> evaluate(const std::vector<float>& x) const;

private:
    std::vector<float> x_knots_;
    std::vector<float> a_coef_;
    std::vector<float> b_coef_;
    std::vector<float> c_coef_;
    std::vector<float> d_coef_;
    std::shared_ptr<ONNXModel> model_;
};
