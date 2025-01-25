package com.splinevalidator;

import com.opencsv.CSVReader;
import com.opencsv.exceptions.CsvValidationException;

import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class Main {
    public static void main(String[] args) {
        // Base directory for spline data
        Path baseDir = Paths.get("spline_data");
        String[] testCases = {
            "cubic_linear", "cubic_quadratic", "cubic_cubic",
            "cubic_seventh_degree", "cubic_sine", "cubic_sine_cosine",
            "cubic_poly7_1", "cubic_poly7_2", "cubic_tanh"
        };

        System.out.println("Running Java validator tests...\n");

        for (String testCase : testCases) {
            try {
                validateSpline(baseDir, testCase);
            } catch (Exception e) {
                System.err.printf("Error testing %s: %s%n", testCase, e.getMessage());
                e.printStackTrace();
            }
        }
    }

    private static void validateSpline(Path baseDir, String testCase) throws IOException, CsvValidationException {
        System.out.printf("Testing %s...%n", testCase);

        // Load ONNX model
        String modelPath = baseDir.resolve(testCase + ".onnx").toString();
        try (ONNXModel model = new ONNXModel(modelPath)) {
            // Read spline data
            double[] x = readCSVColumn(baseDir.resolve(testCase + "_x.csv"));
            double[] xKnots = readCSVColumn(baseDir.resolve(testCase + "_knots.csv"));
            double[] aCoef = readCSVColumn(baseDir.resolve(testCase + "_a.csv"));
            double[] bCoef = readCSVColumn(baseDir.resolve(testCase + "_b.csv"));
            double[] cCoef = readCSVColumn(baseDir.resolve(testCase + "_c.csv"));
            double[] dCoef = readCSVColumn(baseDir.resolve(testCase + "_d.csv"));
            double[] yExpected = readCSVColumn(baseDir.resolve(testCase + "_y.csv"));

            // Create spline and evaluate
            CubicSpline spline = new CubicSpline(xKnots, aCoef, bCoef, cCoef, dCoef, model);
            float[] xFloat = toFloatArray(x);
            float[] yComputed = spline.evaluate(xFloat);

            // Calculate errors
            double maxError = 0.0;
            double sumError = 0.0;
            for (int i = 0; i < yExpected.length; i++) {
                double error = Math.abs(yComputed[i] - yExpected[i]);
                maxError = Math.max(maxError, error);
                sumError += error;
            }
            double meanError = sumError / yExpected.length;

            // Print results
            System.out.println("Validation results:");
            System.out.printf("Max absolute error: %.2e%n", maxError);
            System.out.printf("Mean absolute error: %.2e%n", meanError);
            System.out.println();
        }
    }

    private static double[] readCSVColumn(Path path) throws IOException, CsvValidationException {
        List<Double> values = new ArrayList<>();
        try (CSVReader reader = new CSVReader(new FileReader(path.toFile()))) {
            String[] line;
            while ((line = reader.readNext()) != null) {
                values.add(Double.parseDouble(line[0]));
            }
        }
        return values.stream().mapToDouble(Double::doubleValue).toArray();
    }

    private static float[] toFloatArray(double[] arr) {
        float[] result = new float[arr.length];
        for (int i = 0; i < arr.length; i++) {
            result[i] = (float) arr[i];
        }
        return result;
    }
}
