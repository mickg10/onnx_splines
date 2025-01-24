#include "csv_reader.hpp"
#include "cubic_spline.hpp"
#include "onnx_model.hpp"
#include <iostream>
#include <memory>
#include <cmath>

void validate_spline(
    const std::string& model_path,
    const std::string& input_csv,
    const std::string& output_csv
) {
    // Read input data
    auto input_data = CSVReader::readCSV(input_csv, {"x_knots", "a_coef", "b_coef", "c_coef", "d_coef"});
    auto output_data = CSVReader::readCSV(output_csv, {"x", "y_onnx"});

    // Create model and spline
    auto model = std::make_shared<ONNXModel>(model_path);
    CubicSpline spline(
        input_data["x_knots"],
        input_data["a_coef"],
        input_data["b_coef"],
        input_data["c_coef"],
        input_data["d_coef"],
        model
    );

    // Evaluate spline
    auto y_computed = spline.evaluate(output_data["x"]);

    // Compare results
    const auto& y_reference = output_data["y_onnx"];
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

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <input_csv> <output_csv>\n";
        return 1;
    }

    try {
        validate_spline(argv[1], argv[2], argv[3]);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
