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

## Examples

Run the test script to see comparisons between different implementations:

```bash
python test_splines.py
```

This will generate plots comparing the ONNX implementation with SciPy and pure Python implementations for various test functions.
