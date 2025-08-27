#!/bin/bash

echo "Starting ESP32 Edge AI Pipeline..."

# Create necessary directories
mkdir -p results/models/teacher
mkdir -p results/models/student
mkdir -p results/models/optimized
mkdir -p results/logs
mkdir -p results/figures

# 1. Train Teacher Model
echo "Step 1: Training Teacher Model..."
python teacher_model/train_teacher.py --config config/hyperparameters.yaml

# 2. Train Student Model with Knowledge Distillation
echo "Step 2: Training Student Model with Knowledge Distillation..."
python student_model/train_student_kd.py --config config/hyperparameters.yaml

# 3. Apply Model Optimization
echo "Step 3: Applying Model Optimization..."
python optimization/optimization_pipeline.py --input results/models/student/student_model.h5

# 4. Convert to TensorFlow Lite
echo "Step 4: Converting to TensorFlow Lite..."
python tflite_conversion/convert_to_tflite.py --input results/models/optimized/optimized_model.h5

# 5. Generate C Array for ESP32
echo "Step 5: Generating C Array for ESP32..."
python tflite_conversion/generate_c_array.py --input results/models/optimized/model.tflite

# 6. Run ESP32 Simulation
echo "Step 6: Running ESP32 Performance Simulation..."
python esp32_simulation/inference_simulator.py --model results/models/optimized/model.tflite

# 7. Generate Performance Analysis
echo "Step 7: Generating Performance Analysis..."
python scripts/generate_results.py

echo "Pipeline completed! Check results/ directory for outputs."