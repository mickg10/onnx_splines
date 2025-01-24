#include "spline_validator.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <numeric>

SplineValidator::SplineValidator(const std::string& model_path) 
    : env_(std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "SplineValidator"))
{
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);

    session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), session_options);
}

SplineValidator::~SplineValidator() = default;

std::vector<float> SplineValidator::evaluate(
    const std::vector<float>& x_eval,
    const std::vector<float>& x_knots,
    const std::vector<float>& a_coef,
    const std::vector<float>& b_coef,
    const std::vector<float>& c_coef,
    const std::vector<float>& d_coef
) {
    // Prepare input tensors
    std::vector<int64_t> input_shape = {static_cast<int64_t>(x_eval.size())};
    std::vector<int64_t> knots_shape = {static_cast<int64_t>(x_knots.size())};
    std::vector<int64_t> coef_shape = {static_cast<int64_t>(a_coef.size())};

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<float> x_eval_data(x_eval.begin(), x_eval.end());
    std::vector<float> x_knots_data(x_knots.begin(), x_knots.end());
    std::vector<float> a_coef_data(a_coef.begin(), a_coef.end());
    std::vector<float> b_coef_data(b_coef.begin(), b_coef.end());
    std::vector<float> c_coef_data(c_coef.begin(), c_coef.end());
    std::vector<float> d_coef_data(d_coef.begin(), d_coef.end());

    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info, x_eval_data.data(), x_eval_data.size(), input_shape.data(), input_shape.size()));
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info, x_knots_data.data(), x_knots_data.size(), knots_shape.data(), knots_shape.size()));
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info, a_coef_data.data(), a_coef_data.size(), coef_shape.data(), coef_shape.size()));
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info, b_coef_data.data(), b_coef_data.size(), coef_shape.data(), coef_shape.size()));
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info, c_coef_data.data(), c_coef_data.size(), coef_shape.data(), coef_shape.size()));
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info, d_coef_data.data(), d_coef_data.size(), coef_shape.data(), coef_shape.size()));

    // Run inference
    std::vector<const char*> input_names = {"x", "x_known", "a_coef", "b_coef", "c_coef", "d_coef"};
    std::vector<const char*> output_names = {"y"};

    auto output_tensors = session_->Run(
        Ort::RunOptions{nullptr},
        input_names.data(),
        input_tensors.data(),
        input_tensors.size(),
        output_names.data(),
        output_names.size()
    );

    // Get results
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    size_t output_size = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
    return std::vector<float>(output_data, output_data + output_size);
}

std::vector<std::vector<float>> SplineValidator::readCsv(
    const std::string& filename,
    const std::vector<std::string>& columns
) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::string line;
    std::getline(file, line); // Read header

    // Find column indices
    std::vector<int> col_indices;
    std::stringstream ss(line);
    std::string token;
    int idx = 0;
    std::vector<std::string> header;
    
    while (std::getline(ss, token, ',')) {
        header.push_back(token);
    }

    for (const auto& col : columns) {
        auto it = std::find(header.begin(), header.end(), col);
        if (it == header.end()) {
            throw std::runtime_error("Column not found: " + col);
        }
        col_indices.push_back(std::distance(header.begin(), it));
    }

    // Read data
    std::vector<std::vector<float>> result(columns.size());
    while (std::getline(file, line)) {
        ss.clear();
        ss.str(line);
        std::vector<std::string> row;
        while (std::getline(ss, token, ',')) {
            row.push_back(token);
        }
        
        for (size_t i = 0; i < col_indices.size(); ++i) {
            result[i].push_back(std::stof(row[col_indices[i]]));
        }
    }

    return result;
}

void SplineValidator::validate(
    const std::string& input_csv,
    const std::string& output_csv
) {
    // Read input data
    auto input_data = readCsv(input_csv, {"x_knots", "a_coef", "b_coef", "c_coef", "d_coef"});
    auto output_data = readCsv(output_csv, {"x", "y_onnx"});

    // Evaluate model
    auto y_computed = evaluate(
        output_data[0],  // x_eval
        input_data[0],   // x_knots
        input_data[1],   // a_coef
        input_data[2],   // b_coef
        input_data[3],   // c_coef
        input_data[4]    // d_coef
    );

    // Compare results
    const auto& y_reference = output_data[1];
    if (y_computed.size() != y_reference.size()) {
        throw std::runtime_error("Output size mismatch");
    }

    double max_error = 0.0;
    double mean_error = 0.0;

    for (size_t i = 0; i < y_computed.size(); ++i) {
        double error = std::abs(y_computed[i] - y_reference[i]);
        max_error = std::max(max_error, error);
        mean_error += error;
    }
    mean_error /= y_computed.size();

    std::cout << "Validation results:\n";
    std::cout << "Max absolute error: " << max_error << "\n";
    std::cout << "Mean absolute error: " << mean_error << "\n";
}
