# ONNX Splines

This repository provides an ONNX implementation of cubic splines, with comparisons to SciPy and pure Python implementations.

## Features

- ONNX-based cubic spline interpolation using ONNXScript
- Pure Python implementation of cubic splines
- Comparison with SciPy's implementation
- Visualization of interpolation results and errors
- Test suite with various function types (linear, quadratic, sine, etc.)

## Requirements

- Python 3.12
- ONNX
- ONNXScript
- ONNXRuntime
- NumPy
- SciPy
- Matplotlib

## Installation

```bash
conda create -n onnx_py312 python=3.12
conda activate onnx_py312
pip install onnx onnxscript onnxruntime numpy scipy matplotlib
```

## Usage

```python
from spline_model import create_cubic_spline, evaluate_spline, save_spline_model
import numpy as np

# Create spline model
model = create_cubic_spline.to_model_proto()
save_spline_model(model, 'spline.onnx')

# Example usage
x = np.linspace(-1, 1, 10)  # Known x points
y = x**2  # Known y points (quadratic function)
x_full = np.linspace(-1, 1, 100)  # Points to evaluate

# Get coefficients (you would typically get these from fitting)
a = ...  # coefficient arrays
b = ...
c = ...
d = ...

# Evaluate spline
y_spline = evaluate_spline('spline.onnx', x_full, x[:-1], a, b, c, d)
```

## Building and Testing

### Prerequisites

- Python 3.12 with required packages (see Installation section)
- CMake
- Conan package manager
- Clang compiler with C++20 support

### Installation with Build Tools

```bash
# Install Python dependencies
conda create -n onnx_py312 python=3.12
conda activate onnx_py312
pip install onnx onnxscript onnxruntime numpy scipy matplotlib

# Install build tools
conda install cmake conan clang
```

### Building and Testing

The project includes a Makefile with several targets:

```bash
# Build everything and run all tests
make all

# Build only the C++ validator
make build_cpp

# Generate ONNX model and test data
make generate_data

# Run C++ validator tests
make test_cpp

# Run Python tests with plots
make test_python

# Clean build artifacts
make clean
```

### Project Structure

```
.
├── cpp_validator/        # C++ ONNX model validator
│   ├── include/         # Header files
│   ├── src/            # Source files
│   ├── CMakeLists.txt  # CMake build configuration
│   └── conanfile.txt   # Conan dependencies
├── models/             # Generated ONNX models
├── spline_data/        # Test data and results
├── test_splines.py     # Python test script
└── Makefile           # Build and test automation
```

### Validation

The C++ validator (`test_cpp`) verifies the ONNX model against reference data for multiple test functions:
- Linear function
- Quadratic function
- Cubic function
- Seventh-degree polynomial
- Sine function
- Sine-cosine combination
- Hyperbolic tangent
- Custom polynomial functions

Each test compares the ONNX model output with reference data and reports the maximum and mean absolute errors.

## Examples

Run the test script to see comparisons between different implementations:

```bash
python test_splines.py
```

This will generate plots comparing the ONNX implementation with SciPy and pure Python implementations for various test functions.
