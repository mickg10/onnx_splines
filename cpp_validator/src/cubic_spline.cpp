#include "cubic_spline.hpp"
#include <stdexcept>

CubicSpline::CubicSpline(
    const std::vector<float>& x_knots,
    const std::vector<float>& a_coef,
    const std::vector<float>& b_coef,
    const std::vector<float>& c_coef,
    const std::vector<float>& d_coef,
    std::shared_ptr<ONNXModel> model
)
    : x_knots_(x_knots)
    , a_coef_(a_coef)
    , b_coef_(b_coef)
    , c_coef_(c_coef)
    , d_coef_(d_coef)
    , model_(std::move(model))
{
    // Validate inputs
    if (x_knots.empty()) {
        throw std::invalid_argument("x_knots cannot be empty");
    }

    size_t n = x_knots.size();
    if (a_coef.size() != n || b_coef.size() != n || c_coef.size() != n || d_coef.size() != n) {
        throw std::invalid_argument("Coefficient arrays must have same size as knots array");
    }

    if (!model_) {
        throw std::invalid_argument("Model cannot be null");
    }
}

std::vector<float> CubicSpline::evaluate(const std::vector<float>& x) const {
    std::vector<std::pair<std::string, std::vector<float>>> inputs = {
        {"x", x},
        {"x_known", x_knots_},
        {"a_coef", a_coef_},
        {"b_coef", b_coef_},
        {"c_coef", c_coef_},
        {"d_coef", d_coef_}
    };

    return model_->run(inputs, "y");
}
