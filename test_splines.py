import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import os
from spline_model import create_cubic_spline, save_spline_model, evaluate_spline, PurePythonCubicSpline

def linear(x):
    return x

def quadratic(x):
    return x**2

def cubic(x):
    return x**3

def seventh_degree(x):
    return x**7

def sine(x):
    return np.sin(x)

def sine_cosine(x):
    return np.sin(x) + np.cos(2*x)

def poly7_1(x):
    return x**7 - 3*x**5 + 2*x**3 - x

def poly7_2(x):
    return x**7 - 2*x**6 + 3*x**5 - 2*x**4 + x**3 - 2*x**2 + x - 1

def compare_splines(func_name, func, domain):
    """Compare different spline implementations."""
    # Generate x values
    x = np.linspace(domain[0], domain[1], 10)
    x_full = np.linspace(domain[0], domain[1], 100)
    
    # Get true values
    y_true = func(x_full)
    
    # Get scipy interpolation
    cs = CubicSpline(x, func(x))
    y_scipy = cs(x_full)
    
    # Get pure python interpolation
    pure_spline = PurePythonCubicSpline(x, func(x))
    y_pure = pure_spline(x_full)
    
    # Get coefficients from scipy spline
    a = cs.c[3]
    b = cs.c[2]
    c = cs.c[1]
    d = cs.c[0]
    
    # Get ONNX predictions
    y_onnx = evaluate_spline('/tmp/spline.onnx', x_full, x[:-1], a, b, c, d)
    
    # Calculate errors
    mae_scipy = np.mean(np.abs(y_true - y_scipy))
    mae_pure = np.mean(np.abs(y_true - y_pure))
    mae_onnx = np.mean(np.abs(y_true - y_onnx))
    
    max_error_scipy = np.max(np.abs(y_true - y_scipy))
    max_error_pure = np.max(np.abs(y_true - y_pure))
    max_error_onnx = np.max(np.abs(y_true - y_onnx))
    
    print(f"\nResults for {func_name}:")
    print(f"Max error (SciPy): {max_error_scipy:.2e}")
    print(f"Max error (Pure): {max_error_pure:.2e}")
    print(f"Max error (ONNX): {max_error_onnx:.2e}")
    
    return {
        'x': x_full,
        'y_true': y_true,
        'y_scipy': y_scipy,
        'y_pure': y_pure,
        'y_onnx': y_onnx,
        'func_name': func_name
    }

def main():
    """Run the main comparison."""
    print("Running spline comparisons...")
    print("\nMean Absolute Errors:")
    print("-" * 80)
    print(f"{'Function':<14}{'SciPy MAE':<12}{'Pure MAE':<12}{'ONNX MAE':<12}")
    print("-" * 80)
    
    # Create a single figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # Define test functions and their domains
    test_cases = [
        ('linear', linear, (-1, 1)),
        ('quadratic', quadratic, (-1, 1)),
        ('cubic', cubic, (-1, 1)),
        ('7th_degree', seventh_degree, (-1, 1)),
        ('sine', sine, (0, 2*np.pi)),
        ('sine_cosine', sine_cosine, (0, 2*np.pi)),
        ('poly7_1', poly7_1, (-1, 1)),
        ('poly7_2', poly7_2, (-1, 1))
    ]
    
    # Create subplots
    for i, (func_name, func, domain) in enumerate(test_cases, 1):
        results = compare_splines(func_name, func, domain)
        
        ax1 = fig.add_subplot(3, 3, i)
        
        # Plot splines on left axis
        ax1.plot(results['x'], results['y_true'], 'k-', label='True', alpha=0.5)
        ax1.plot(results['x'], results['y_scipy'], 'r--', label='SciPy')
        ax1.plot(results['x'], results['y_pure'], 'g:', label='Pure Python')
        ax1.plot(results['x'], results['y_onnx'], 'b-.', label='ONNX')
        ax1.set_title(results['func_name'])
        ax1.grid(True)
        
        # Create right axis for errors
        ax2 = ax1.twinx()
        
        # Calculate and plot errors
        error_scipy = np.abs(results['y_true'] - results['y_scipy'])
        error_pure = np.abs(results['y_true'] - results['y_pure'])
        error_onnx = np.abs(results['y_true'] - results['y_onnx'])
        
        ax2.plot(results['x'], error_scipy, 'r:', alpha=0.3, label='SciPy Error')
        ax2.plot(results['x'], error_pure, 'g:', alpha=0.3, label='Pure Error')
        ax2.plot(results['x'], error_onnx, 'b:', alpha=0.3, label='ONNX Error')
        ax2.set_yscale('log')
        ax2.tick_params(axis='y', labelcolor='gray')
        
        if i == 1:  # Only show legends for the first subplot
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.savefig('spline_comparison.png')
    plt.show()
    print("-" * 80)
    print("\nPlots have been saved as 'spline_comparison.png'")

if __name__ == "__main__":
    # Create and save ONNX model
    model = create_cubic_spline.to_model_proto()
    save_spline_model(model, '/tmp/spline.onnx')
    
    main()
