cmake_minimum_required(VERSION 3.15)
project(spline_validator)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(onnxruntime REQUIRED)
find_package(Boost REQUIRED)

add_executable(spline_validator
    src/main.cpp
    src/onnx_model.cpp
    src/cubic_spline.cpp
    src/csv_reader.cpp
)

target_include_directories(spline_validator PRIVATE include)
target_link_libraries(spline_validator PRIVATE
    onnxruntime::onnxruntime
    Boost::boost
)
