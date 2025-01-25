.PHONY: all build_cpp test_cpp generate_data test_python clean build_java test_java

# Default conda environment
CONDA_DIR := /Users/mickg10/miniconda3
CONDA_ENV := onnx_py311
CONDA_ACTIVATE := source $(CONDA_DIR)/etc/profile.d/conda.sh && conda activate $(CONDA_ENV)

# Directories
CPP_DIR := cpp_validator
JAVA_DIR := java_validator
BUILD_DIR := $(CPP_DIR)/build
MODELS_DIR := models
DATA_DIR := spline_data

all: build_cpp build_java generate_data test_cpp test_java test_python

build_cpp:
	@echo "Building C++ validator..."
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && \
	$(CONDA_ACTIVATE) && \
	conan install .. --output-folder=. --build=missing -s compiler.cppstd=20 && \
	cmake .. -DCMAKE_TOOLCHAIN_FILE=build/Release/generators/conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Release && \
	cmake --build .

test_cpp: build_cpp generate_data
	@echo "Running C++ validator tests..."
	@cd $(BUILD_DIR) && \
	for input in $(CURDIR)/$(DATA_DIR)/cubic_*_input.csv; do \
		output=$${input/_input.csv/_output.csv}; \
		echo "\nTesting $$(basename $${input/_input.csv/})..."; \
		./spline_validator \
			$(CURDIR)/$(MODELS_DIR)/spline.onnx \
			$$input \
			$$output || exit 1; \
	done

generate_data:
	@echo "Generating ONNX model and test data..."
	@mkdir -p $(MODELS_DIR)
	@mkdir -p $(DATA_DIR)
	@$(CONDA_ACTIVATE) && \
	python test_splines.py \
		--model-path $(MODELS_DIR)/spline.onnx \
		--data-dir $(DATA_DIR) \
		--no-plot

test_python:
	@echo "Running Python tests with plots..."
	@$(CONDA_ACTIVATE) && \
	python test_splines.py \
		--model-path $(MODELS_DIR)/spline.onnx \
		--data-dir $(DATA_DIR)

build_java:
	@echo "Building Java validator..."
	@cd $(JAVA_DIR) && ./gradlew build

test_java: build_java generate_data
	@echo "Running Java validator tests..."
	@cd $(JAVA_DIR) && ./gradlew test

clean:
	@echo "Cleaning build artifacts..."
	@rm -rf $(BUILD_DIR)
	@rm -rf $(MODELS_DIR)
	@rm -rf $(DATA_DIR)
	@rm -f spline_comparison.png
	@cd $(JAVA_DIR) && ./gradlew clean
	@echo "Clean complete"
