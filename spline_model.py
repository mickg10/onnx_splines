import numpy as np
import onnx
import onnxruntime as ort
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset20 as op


class PurePythonCubicSpline:
    """Pure Python implementation of cubic spline interpolation."""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.n = len(x)
        
        h = np.diff(x)
        dy = np.diff(y)
        
        # Get linear system coefficients
        A = np.zeros((self.n, self.n))
        b = np.zeros(self.n)
        
        # Interior points
        for i in range(1, self.n-1):
            A[i, i-1:i+2] = [h[i-1], 2*(h[i-1] + h[i]), h[i]]
            b[i] = 3*(dy[i]/h[i] - dy[i-1]/h[i-1])
        
        # Boundary conditions (natural spline)
        A[0, 0] = 1
        A[-1, -1] = 1
        
        # Solve for c coefficients
        self.c = np.linalg.solve(A, b)
        
        # Get remaining coefficients
        self.a = y[:-1]
        self.b = dy/h - h*(2*self.c[:-1] + self.c[1:])/3
        self.d = (self.c[1:] - self.c[:-1])/(3*h)
    
    def __call__(self, x_new):
        """Evaluate the spline at new points."""
        y_new = np.zeros_like(x_new)
        
        for i, x in enumerate(x_new):
            # Find the appropriate interval
            idx = np.searchsorted(self.x, x) - 1
            idx = np.clip(idx, 0, self.n-2)
            
            # Calculate dx
            dx = x - self.x[idx]
            
            # Evaluate cubic polynomial
            y_new[i] = self.a[idx] + self.b[idx]*dx + self.c[idx]*dx**2 + self.d[idx]*dx**3
        
        return y_new


@script()
def create_cubic_spline(x: FLOAT["N"], x_known: FLOAT["M"], 
                       a_coef: FLOAT["M"], b_coef: FLOAT["M"], 
                       c_coef: FLOAT["M"], d_coef: FLOAT["M"]) -> FLOAT["N"]:
    """Create a cubic spline model."""
    # Reshape x to 2D for broadcasting
    x_unsqueezed = op.Unsqueeze(x, axes=[1])
    
    # Calculate absolute difference between x and x_known
    x_diff = op.Sub(x_unsqueezed, x_known)
    x_diff_abs = op.Abs(x_diff)
    
    # Find index of closest x value using argmin
    indices = op.ArgMin(x_diff_abs, axis=1, keepdims=0)
    
    # Clip indices to valid range
    n = op.Shape(x_known)
    n_minus_2 = op.Sub(n, op.Constant(value_ints=[2]))
    zero = op.Constant(value_ints=[0])
    indices_clipped = op.Clip(indices, zero, n_minus_2)
    
    # Gather coefficients
    x_start = op.Gather(x_known, indices_clipped)
    a = op.Gather(a_coef, indices_clipped)
    b = op.Gather(b_coef, indices_clipped)
    c = op.Gather(c_coef, indices_clipped)
    d = op.Gather(d_coef, indices_clipped)
    
    # Calculate dx
    dx = op.Sub(x, x_start)
    
    # Calculate polynomial value: a + b*dx + c*dx^2 + d*dx^3
    dx_squared = op.Mul(dx, dx)
    dx_cubed = op.Mul(dx_squared, dx)
    
    c_term = op.Mul(c, dx_squared)
    d_term = op.Mul(d, dx_cubed)
    b_term = op.Mul(b, dx)
    
    # Sum all terms
    y = op.Add(op.Add(op.Add(a, b_term), c_term), d_term)
    
    return y

def evaluate_spline(model_path: str, x_full: np.ndarray, x: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Evaluate the spline using ONNX Runtime."""
    # Convert inputs to float32
    x = x.astype(np.float32)
    x_full = x_full.astype(np.float32)
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    c = c.astype(np.float32)
    d = d.astype(np.float32)
    
    # Create session
    session = ort.InferenceSession(model_path)
    
    # Prepare inputs
    inputs = {
        'x': x_full,
        'x_known': x,
        'a_coef': a,
        'b_coef': b,
        'c_coef': c,
        'd_coef': d
    }
    
    # Run inference
    outputs = session.run(None, inputs)
    return outputs[0]

def save_spline_model(model: onnx.ModelProto, filepath: str) -> None:
    """Save an ONNX model to a file."""
    onnx.save(model, filepath)
