#pragma once

#include <string>
#include <vector>
#include <memory>
#include <onnxruntime_cxx_api.h>

class SplineValidator {
public:
    SplineValidator(const std::string& model_path);
    ~SplineValidator();

    // Evaluate the spline for given inputs
    std::vector<float> evaluate(
        const std::vector<float>& x_eval,
        const std::vector<float>& x_knots,
        const std::vector<float>& a_coef,
        const std::vector<float>& b_coef,
        const std::vector<float>& c_coef,
        const std::vector<float>& d_coef
    );

    // Compare results with reference data
    void validate(
        const std::string& input_csv,
        const std::string& output_csv
    );

private:
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    Ort::AllocatorWithDefaultOptions allocator_;

    // Helper to read CSV files
    std::vector<std::vector<float>> readCsv(const std::string& filename, const std::vector<std::string>& columns);
};
