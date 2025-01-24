#pragma once

#include <string>
#include <vector>
#include <memory>

namespace Ort {
    class Session;
    class Env;
}

class ONNXModel {
public:
    explicit ONNXModel(const std::string& model_path);
    ~ONNXModel();

    // Prevent copying
    ONNXModel(const ONNXModel&) = delete;
    ONNXModel& operator=(const ONNXModel&) = delete;

    // Allow moving
    ONNXModel(ONNXModel&&) noexcept;
    ONNXModel& operator=(ONNXModel&&) noexcept;

    // Run inference
    std::vector<float> run(
        const std::vector<std::pair<std::string, std::vector<float>>>& inputs,
        const std::string& output_name
    ) const;

private:
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
};
