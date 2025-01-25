package com.splinevalidator;

import org.apache.commons.math3.analysis.interpolation.SplineInterpolator;
import org.apache.commons.math3.analysis.polynomials.PolynomialSplineFunction;

public class CubicSpline {
    private final double[] xKnots;
    private final double[] aCoef;
    private final double[] bCoef;
    private final double[] cCoef;
    private final double[] dCoef;
    private final ONNXModel model;

    public CubicSpline(double[] xKnots, double[] aCoef, double[] bCoef, double[] cCoef, double[] dCoef, ONNXModel model) {
        if (xKnots == null || xKnots.length == 0) {
            throw new IllegalArgumentException("xKnots cannot be null or empty");
        }

        int n = xKnots.length;
        if (aCoef.length != n || bCoef.length != n || cCoef.length != n || dCoef.length != n) {
            throw new IllegalArgumentException("Coefficient arrays must have same size as knots array");
        }

        if (model == null) {
            throw new IllegalArgumentException("Model cannot be null");
        }

        this.xKnots = xKnots.clone();
        this.aCoef = aCoef.clone();
        this.bCoef = bCoef.clone();
        this.cCoef = cCoef.clone();
        this.dCoef = dCoef.clone();
        this.model = model;
    }

    public float[] evaluate(float[] x) {
        // Convert inputs to float arrays for ONNX
        float[] xKnotsFloat = new float[xKnots.length];
        float[] aCoefFloat = new float[aCoef.length];
        float[] bCoefFloat = new float[bCoef.length];
        float[] cCoefFloat = new float[cCoef.length];
        float[] dCoefFloat = new float[dCoef.length];

        // Convert all inputs to float and ensure proper ordering
        for (int i = 0; i < xKnots.length; i++) {
            xKnotsFloat[i] = (float) xKnots[i];
            aCoefFloat[i] = (float) aCoef[i];
            bCoefFloat[i] = (float) bCoef[i];
            cCoefFloat[i] = (float) cCoef[i];
            dCoefFloat[i] = (float) dCoef[i];
        }

        // Run the model with preprocessed inputs
        return model.run(x, xKnotsFloat, aCoefFloat, bCoefFloat, cCoefFloat, dCoefFloat);
    }
}
