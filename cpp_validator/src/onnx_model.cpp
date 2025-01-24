#include "onnx_model.hpp"
#include <onnxruntime_cxx_api.h>
#include <numeric>

ONNXModel::ONNXModel(const std::string& model_path)
    : env_(std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "spline_validator"))
{
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);

    session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), session_options);
}

ONNXModel::~ONNXModel() = default;

ONNXModel::ONNXModel(ONNXModel&&) noexcept = default;
ONNXModel& ONNXModel::operator=(ONNXModel&&) noexcept = default;

std::vector<float> ONNXModel::run(
    const std::vector<std::pair<std::string, std::vector<float>>>& inputs,
    const std::string& output_name
) const {
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator,
        OrtMemType::OrtMemTypeDefault
    );

    // Prepare input tensors
    std::vector<Ort::Value> input_tensors;
    std::vector<const char*> input_names;

    for (const auto& [name, data] : inputs) {
        std::vector<int64_t> input_shape{static_cast<int64_t>(data.size())};
        input_tensors.push_back(
            Ort::Value::CreateTensor<float>(
                memory_info,
                const_cast<float*>(data.data()),
                data.size(),
                input_shape.data(),
                input_shape.size()
            )
        );
        input_names.push_back(name.c_str());
    }

    // Run inference
    const char* output_names[] = {output_name.c_str()};
    auto output_tensors = session_->Run(
        Ort::RunOptions{nullptr},
        input_names.data(),
        input_tensors.data(),
        input_tensors.size(),
        output_names,
        1
    );

    // Get output data
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    size_t output_size = std::accumulate(output_shape.begin(), output_shape.end(), 1LL, std::multiplies<int64_t>());

    return std::vector<float>(output_data, output_data + output_size);
}
