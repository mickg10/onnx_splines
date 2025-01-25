package com.splinevalidator;

import ai.onnxruntime.*;

import java.nio.FloatBuffer;
import java.util.HashMap;
import java.util.Map;

public class ONNXModel implements AutoCloseable {
    private final OrtEnvironment env;
    private final OrtSession session;

    public ONNXModel(String modelPath) {
        try {
            env = OrtEnvironment.getEnvironment();
            session = env.createSession(modelPath);
        } catch (OrtException e) {
            throw new RuntimeException("Failed to load ONNX model", e);
        }
    }

    public float[] run(float[] x, float[] xKnown, float[] aCoef, float[] bCoef, float[] cCoef, float[] dCoef) {
        try {
            Map<String, OnnxTensor> inputs = new HashMap<>();
            long[] xShape = new long[]{x.length};
            long[] knotShape = new long[]{xKnown.length};

            // Create tensors for all inputs
            FloatBuffer xBuffer = FloatBuffer.wrap(x);
            FloatBuffer xKnownBuffer = FloatBuffer.wrap(xKnown);
            FloatBuffer aCoefBuffer = FloatBuffer.wrap(aCoef);
            FloatBuffer bCoefBuffer = FloatBuffer.wrap(bCoef);
            FloatBuffer cCoefBuffer = FloatBuffer.wrap(cCoef);
            FloatBuffer dCoefBuffer = FloatBuffer.wrap(dCoef);

            // Add inputs to the map
            inputs.put("x", OnnxTensor.createTensor(env, xBuffer, xShape));
            inputs.put("x_known", OnnxTensor.createTensor(env, xKnownBuffer, knotShape));
            inputs.put("a_coef", OnnxTensor.createTensor(env, aCoefBuffer, knotShape));
            inputs.put("b_coef", OnnxTensor.createTensor(env, bCoefBuffer, knotShape));
            inputs.put("c_coef", OnnxTensor.createTensor(env, cCoefBuffer, knotShape));
            inputs.put("d_coef", OnnxTensor.createTensor(env, dCoefBuffer, knotShape));

            // Run the model
            try (OrtSession.Result results = session.run(inputs)) {
                // Get the output tensor
                OnnxTensor output = (OnnxTensor) results.get(0);
                
                // Convert output to float array
                FloatBuffer outputBuffer = FloatBuffer.allocate(x.length);
                output.getFloatBuffer().get(outputBuffer.array());
                return outputBuffer.array();
            }
        } catch (OrtException e) {
            throw new RuntimeException("Failed to run ONNX model", e);
        }
    }

    @Override
    public void close() {
        try {
            session.close();
            env.close();
        } catch (OrtException e) {
            throw new RuntimeException("Failed to close ONNX resources", e);
        }
    }
}
