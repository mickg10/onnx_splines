import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import os
from spline_model import create_cubic_spline, save_spline_model, evaluate_spline, PurePythonCubicSpline
import onnxruntime
import onnx
import argparse
import pandas as pd

print(f"ONNX Runtime Version: {onnxruntime.__version__}")
print(f"ONNX IR Version: {onnx.__version__}")
print("\nRunning spline comparisons...")

def cubic_linear(x):
    return x

def cubic_quadratic(x):
    return x**2

def cubic_cubic(x):
    return x**3

def cubic_seventh_degree(x):
    return x**7

def cubic_sine(x):
    return np.sin(x)

def cubic_sine_cosine(x):
    return np.sin(x) + np.cos(2*x)

def cubic_poly7_1(x):
    return x**7 - 3*x**5 + 2*x**3 - x

def cubic_poly7_2(x):
    return x**7 + 5*x**6 - 2*x**4 + 3*x**2

def cubic_tanh_func(x):
    return np.tanh(x)

def save_spline_input_data(x_knots, y_knots, a_coef, b_coef, c_coef, d_coef, filename):
    """Save spline input data (knots and coefficients) to CSV."""
    df = pd.DataFrame({
        'x_knots': x_knots,
        'y_knots': y_knots,
        'a_coef': a_coef,
        'b_coef': b_coef,
        'c_coef': c_coef,
        'd_coef': d_coef
    })
    df.to_csv(filename, index=False)

def save_evaluation_data(x_eval, y_true, y_scipy, y_pure, y_onnx, filename):
    """Save evaluation results to CSV."""
    df = pd.DataFrame({
        'x': x_eval,
        'y_true': y_true,
        'y_scipy': y_scipy,
        'y_pure': y_pure,
        'y_onnx': y_onnx,
        'error_scipy': y_scipy - y_true,
        'error_pure': y_pure - y_true,
        'error_onnx': y_onnx - y_true
    })
    df.to_csv(filename, index=False)

def plot_all_spline_comparisons(test_cases, results):
    """Plot all spline comparisons in a single figure."""
    n_funcs = len(test_cases)
    fig = plt.figure(figsize=(20, 4*n_funcs))
    gs = plt.GridSpec(n_funcs, 1, figure=fig, hspace=0.4)
    
    for idx, (func_name, x, x_full, y, y_true, y_scipy, y_pure, y_onnx) in enumerate(results):
        # Calculate derivatives
        dx = x_full[1] - x_full[0]
        dy_true = np.gradient(y_true, dx)
        dy_scipy = np.gradient(y_scipy, dx)
        dy_pure = np.gradient(y_pure, dx)
        dy_onnx = np.gradient(y_onnx, dx)
        
        # Calculate differences (errors)
        diff_scipy = y_scipy - y_true
        diff_pure = y_pure - y_true
        diff_onnx = y_onnx - y_true
        
        # Create subplot with twin axis
        ax1 = fig.add_subplot(gs[idx])
        ax2 = ax1.twinx()
        
        # Plot values and derivatives on left axis
        lines1 = []
        # Plot true function and spline fits
        lines1.extend(ax1.plot(x_full, y_true, 'k-', label='True', linewidth=2))
        lines1.extend(ax1.plot(x_full, y_scipy, 'r--', label='SciPy', alpha=0.7))
        lines1.extend(ax1.plot(x_full, y_pure, 'g--', label='Pure Python', alpha=0.7))
        lines1.extend(ax1.plot(x_full, y_onnx, 'b--', label='ONNX', alpha=0.7))
        
        # Plot knot points
        lines1.extend(ax1.plot(x, y, 'ko', label='Knot Points', markersize=6))
        
        # Plot derivatives with thinner lines and different style
        lines1.extend(ax1.plot(x_full, dy_true, 'k:', label='True d/dx', linewidth=1, alpha=0.5))
        lines1.extend(ax1.plot(x_full, dy_scipy, 'r:', label='SciPy d/dx', alpha=0.5, linewidth=1))
        lines1.extend(ax1.plot(x_full, dy_pure, 'g:', label='Pure d/dx', alpha=0.5, linewidth=1))
        lines1.extend(ax1.plot(x_full, dy_onnx, 'b:', label='ONNX d/dx', alpha=0.5, linewidth=1))
        
        # Plot differences on right axis with markers to make them more visible
        lines2 = []
        lines2.extend(ax2.plot(x_full, diff_scipy, 'r-.', label='SciPy Error', alpha=0.7, marker='.', markersize=2, markevery=10))
        lines2.extend(ax2.plot(x_full, diff_pure, 'g-.', label='Pure Error', alpha=0.7, marker='.', markersize=2, markevery=10))
        lines2.extend(ax2.plot(x_full, diff_onnx, 'b-.', label='ONNX Error', alpha=0.7, marker='.', markersize=2, markevery=10))
        
        # Add horizontal line at y=0 for error axis
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.2)
        
        # Set labels and title
        ax1.set_title(f'{func_name} - Function Values, Derivatives, and Errors', pad=20)
        ax1.set_xlabel('x')
        ax1.set_ylabel('Value / Derivative')
        ax2.set_ylabel('Error (y - y_true)')
        
        # Add grid
        ax1.grid(True, alpha=0.3)
        
        # Combine legends
        lines = lines1 + lines2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left', bbox_to_anchor=(1.15, 1.0))
        
    plt.tight_layout()
    plt.savefig('spline_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Run the main comparison."""
    parser = argparse.ArgumentParser(description='Test spline implementations and save results')
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting results')
    parser.add_argument('--model-path', type=str, default='/tmp/spline.onnx', help='Path to save ONNX model')
    parser.add_argument('--data-dir', type=str, default='data', help='Directory to save CSV files')
    args = parser.parse_args()

    # Create data directory if it doesn't exist
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    print("\nMean Absolute Errors:")
    print("-" * 80)
    print(f"{'Function':<14}{'SciPy MAE':<12}{'Pure MAE':<12}{'ONNX MAE':<12}")
    print("-" * 80)
    
    # Create and save ONNX model
    model = create_cubic_spline.to_model_proto()
    save_spline_model(model, args.model_path)
    
    # Define test functions and their domains
    test_cases = [
        ('cubic_linear', cubic_linear, (-1, 1)),
        ('cubic_quadratic', cubic_quadratic, (-1, 1)),
        ('cubic_cubic', cubic_cubic, (-1, 1)),
        ('cubic_seventh_degree', cubic_seventh_degree, (-1, 1)),
        ('cubic_sine', cubic_sine, (0, 2*np.pi)),
        ('cubic_sine_cosine', cubic_sine_cosine, (0, 2*np.pi)),
        ('cubic_poly7_1', cubic_poly7_1, (-1, 1)),
        ('cubic_poly7_2', cubic_poly7_2, (-1, 1)),
        ('cubic_tanh', cubic_tanh_func, (-2, 2))
    ]

    results = []
    for func_name, func, domain in test_cases:
        # Generate data
        x = np.linspace(domain[0], domain[1], 20)  # Fewer points for fitting
        x_full = np.linspace(domain[0], domain[1], 1000)  # More points for evaluation
        y = func(x)
        y_true = func(x_full)
        
        # Fit splines
        scipy_spline = CubicSpline(x, y)
        pure_spline = PurePythonCubicSpline(x, y)
        
        # Get coefficients from scipy spline for ONNX
        a = scipy_spline.c[3]
        b = scipy_spline.c[2]
        c = scipy_spline.c[1]
        d = scipy_spline.c[0]
        
        # Create and evaluate splines
        y_scipy = scipy_spline(x_full)
        y_pure = pure_spline(x_full)
        y_onnx = evaluate_spline(args.model_path, x_full, x[:-1], a, b, c, d)

        # Save input and output data
        save_spline_input_data(x[:-1], y[:-1], a, b, c, d, 
                             os.path.join(args.data_dir, f'{func_name}_input.csv'))
        save_evaluation_data(x_full, y_true, y_scipy, y_pure, y_onnx,
                           os.path.join(args.data_dir, f'{func_name}_output.csv'))

        # Store results for plotting
        results.append((func_name, x, x_full, y, y_true, y_scipy, y_pure, y_onnx))

        # Print errors
        diff_scipy = y_scipy - y_true
        diff_pure = y_pure - y_true
        diff_onnx = y_onnx - y_true
        print(f"\nResults for {func_name}:")
        print(f"Max error (SciPy): {np.max(np.abs(diff_scipy)):.2e}")
        print(f"Max error (Pure): {np.max(np.abs(diff_pure)):.2e}")
        print(f"Max error (ONNX): {np.max(np.abs(diff_onnx)):.2e}")

    if not args.no_plot:
        plot_all_spline_comparisons(test_cases, results)

if __name__ == "__main__":
    main()
