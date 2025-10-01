"""
EloquentTinyML Model Converter for esp32
Converts TensorFlow Lite models to EloquentTinyML compatible format
"""

import os
import numpy as np
import tensorflow as tf
from pathlib import Path

class EloquentTinyMLConverter:
    def __init__(self, models_dir='models'):
        self.models_dir = Path(models_dir)
        self.output_dir = self.models_dir / 'eloquent'
        self.output_dir.mkdir(exist_ok=True)
        
    def convert_tflite_to_eloquent(self, tflite_path, model_name):
        """Convert a TFLite model to EloquentTinyML format"""
        print(f"Converting {model_name} model...")
        
        # Read the TFLite model
        with open(tflite_path, 'rb') as f:
            tflite_model = f.read()
        
        # Generate the C header file in EloquentTinyML format
        header_content = self.generate_eloquent_header(tflite_model, model_name)
        
        # Write the header file
        header_path = self.output_dir / f"{model_name}_model.h"
        with open(header_path, 'w') as f:
            f.write(header_content)
        
        print(f"✅ Generated: {header_path}")
        return header_path
    
    def generate_eloquent_header(self, tflite_model, model_name):
        """Generate EloquentTinyML compatible header"""
        model_size = len(tflite_model)
        
        header = f"""#pragma once
#include <pgmspace.h>

/**
 * {model_name.upper()} TensorFlow Lite model for EloquentTinyML
 * Model size: {model_size} bytes
 * Generated for esp32 deployment
 */

// Model data in PROGMEM
const unsigned char {model_name}_model[] PROGMEM = {{
"""
        
        # Convert bytes to hex array
        hex_array = []
        for i, byte in enumerate(tflite_model):
            if i % 12 == 0:
                hex_array.append('\n    ')
            hex_array.append(f'0x{byte:02x}')
            if i < len(tflite_model) - 1:
                hex_array.append(', ')
        
        header += ''.join(hex_array)
        header += f"""
}};

// Model metadata
const unsigned int {model_name}_model_len = {model_size};

"""
        return header
    
    def create_test_samples_header(self):
        """Create test samples header for all datasets"""
        header = """#pragma once

/**
 * Test samples for esp32 Edge AI validation
 */

// MNIST test sample (flattened 28x28 image)
const float mnist_test_sample[] PROGMEM = {
    // Sample for digit '7' - you can replace with actual test data
"""
        
        # Generate a simple test pattern for MNIST (784 values)
        # In practice, you'd use actual test data from your training
        mnist_sample = np.random.rand(784) * 0.5  # Simple pattern
        for i, val in enumerate(mnist_sample):
            if i % 8 == 0:
                header += "\n    "
            header += f"{val:.6f}f"
            if i < len(mnist_sample) - 1:
                header += ", "
        
        header += """
};
const int mnist_expected_output = 7;

// Iris test sample (4 features)
const float iris_test_sample[] PROGMEM = {
    5.1f, 3.5f, 1.4f, 0.2f  // Setosa sample
};
const int iris_expected_output = 0;

// Sensor test sample (4 sensor readings)
const float sensor_test_sample[] PROGMEM = {
    25.5f, 65.2f, 1013.2f, 9.8f  // Normal conditions
};
const int sensor_expected_output = 0;

"""
        
        # Write the test samples header
        header_path = self.output_dir / "test_samples.h"
        with open(header_path, 'w') as f:
            f.write(header)
        
        print(f"✅ Generated: {header_path}")
        return header_path
    
    def create_arduino_sketch(self):
        """Create a complete Arduino sketch for esp32"""
        sketch_content = '''/*
 * esp32 Edge AI - EloquentTinyML Multi-Dataset Inference
 * Compatible with esp32 NodeMCU (80MHz, 80KB RAM)
 */

#include <Arduino.h>
#include <eloquent_tinyml.h>

// Model headers
#include "mnist_model.h"
#include "iris_model.h"
#include "sensor_model.h"
#include "test_samples.h"

// Create TensorFlow instances with memory allocation for esp32
Eloquent::TinyML::TensorFlow::TensorFlow<8000> mnist_tf;
Eloquent::TinyML::TensorFlow::TensorFlow<4000> iris_tf;
Eloquent::TinyML::TensorFlow::TensorFlow<6000> sensor_tf;

void setup() {
    Serial.begin(115200);
    delay(2000);
    
    Serial.println("esp32 Edge AI - EloquentTinyML");
    Serial.println("================================");
    Serial.printf("Free heap: %d bytes\\n", ESP.getFreeHeap());
    
    // Initialize models
    Serial.println("Initializing models...");
    
    if (mnist_tf.begin(mnist_model)) {
        Serial.println("✅ MNIST model loaded");
    } else {
        Serial.println("❌ MNIST model failed");
    }
    
    if (iris_tf.begin(iris_model)) {
        Serial.println("✅ Iris model loaded");
    } else {
        Serial.println("❌ Iris model failed");
    }
    
    if (sensor_tf.begin(sensor_model)) {
        Serial.println("✅ Sensor model loaded");
    } else {
        Serial.println("❌ Sensor model failed");
    }
    
    Serial.println("\\nRunning inference tests...");
    runTests();
}

void runTests() {
    // Test MNIST
    Serial.println("\\n--- MNIST Test ---");
    unsigned long start = micros();
    float mnist_result = mnist_tf.predict((float*)mnist_test_sample);
    unsigned long elapsed = micros() - start;
    
    Serial.printf("Prediction: %.3f, Expected: %d\\n", mnist_result, mnist_expected_output);
    Serial.printf("Inference time: %lu μs (%.2f ms)\\n", elapsed, elapsed/1000.0);
    
    // Test Iris
    Serial.println("\\n--- Iris Test ---");
    start = micros();
    float iris_result = iris_tf.predict((float*)iris_test_sample);
    elapsed = micros() - start;
    
    Serial.printf("Prediction: %.3f, Expected: %d\\n", iris_result, iris_expected_output);
    Serial.printf("Inference time: %lu μs (%.2f ms)\\n", elapsed, elapsed/1000.0);
    
    // Test Sensor
    Serial.println("\\n--- Sensor Test ---");
    start = micros();
    float sensor_result = sensor_tf.predict((float*)sensor_test_sample);
    elapsed = micros() - start;
    
    Serial.printf("Prediction: %.3f, Expected: %d\\n", sensor_result, sensor_expected_output);
    Serial.printf("Inference time: %lu μs (%.2f ms)\\n", elapsed, elapsed/1000.0);
    
    Serial.printf("\\nFree heap after inference: %d bytes\\n", ESP.getFreeHeap());
}

void loop() {
    // Heartbeat LED
    static unsigned long lastBlink = 0;
    if (millis() - lastBlink > 1000) {
        digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN));
        lastBlink = millis();
    }
    delay(100);
}
'''
        
        sketch_path = self.output_dir / "esp32_eloquent_inference.ino"
        with open(sketch_path, 'w') as f:
            f.write(sketch_content)
        
        print(f"✅ Generated Arduino sketch: {sketch_path}")
        return sketch_path
    
    def convert_all_models(self):
        """Convert all available TFLite models"""
        model_files = {
            'mnist': 'mnist_optimized.tflite',
            'iris': 'iris_optimized.tflite', 
            'sensor': 'sensor_optimized.tflite',
            'fashion_mnist': 'fashion_mnist_optimized.tflite'
        }
        
        converted_models = []
        
        for model_name, filename in model_files.items():
            tflite_path = self.models_dir / filename
            
            if tflite_path.exists():
                try:
                    header_path = self.convert_tflite_to_eloquent(tflite_path, model_name)
                    converted_models.append((model_name, header_path))
                except Exception as e:
                    print(f"❌ Failed to convert {model_name}: {e}")
            else:
                print(f"⚠️  Model file not found: {tflite_path}")
        
        return converted_models
    
    def create_installation_guide(self):
        """Create installation and usage guide"""
        guide = """# esp32 EloquentTinyML Setup Guide

## Prerequisites

1. Install Arduino IDE
2. Install esp32 board package:
   - File → Preferences → Additional Board Manager URLs
   - Add: http://arduino.esp32.com/stable/package_esp32com_index.json
   - Tools → Board → Boards Manager → Search "esp32" → Install

3. Install EloquentTinyML library:
   - Tools → Manage Libraries → Search "EloquentTinyML" → Install

## Hardware Setup

- esp32 NodeMCU v1.0
- USB cable
- Computer with Arduino IDE

## File Structure

```
eloquent/
├── mnist_model.h           # MNIST model header
├── iris_model.h            # Iris model header  
├── sensor_model.h          # Sensor model header
├── test_samples.h          # Test data samples
├── esp32_eloquent_inference.ino  # Main Arduino sketch
└── README.md               # This guide
```

## Usage

1. Copy all .h files to your Arduino sketch folder
2. Open esp32_eloquent_inference.ino in Arduino IDE
3. Select board: NodeMCU 1.0 (ESP-12E Module)
4. Select correct COM port
5. Upload the sketch
6. Open Serial Monitor (115200 baud)
7. Observe inference results

## Memory Configuration

The sketch allocates:
- MNIST model: 8000 bytes
- Iris model: 4000 bytes  
- Sensor model: 6000 bytes
- Total: ~18KB (within esp32's 80KB RAM limit)

## Troubleshooting

1. **Compilation errors**: Ensure all .h files are in sketch folder
2. **Upload failures**: Check COM port and board selection
3. **Runtime errors**: Monitor Serial output for heap usage
4. **Poor performance**: Reduce model memory allocation if needed

## Performance Expectations

- MNIST: ~50-100ms inference time
- Iris: ~20-50ms inference time
- Sensor: ~30-70ms inference time

Memory usage should remain stable around 60-70KB free heap.
"""
        
        guide_path = self.output_dir / "README.md"
        with open(guide_path, 'w') as f:
            f.write(guide)
        
        print(f"✅ Generated setup guide: {guide_path}")
        return guide_path

def main():
    """Main conversion process"""
    print("EloquentTinyML Model Converter for esp32")
    print("=" * 50)
    
    converter = EloquentTinyMLConverter()
    
    # Convert all available models
    print("Converting TFLite models to EloquentTinyML format...")
    converted_models = converter.convert_all_models()
    
    # Create test samples
    print("\nGenerating test samples...")
    converter.create_test_samples_header()
    
    # Create Arduino sketch
    print("\nGenerating Arduino sketch...")
    converter.create_arduino_sketch()
    
    # Create installation guide
    print("\nGenerating setup guide...")
    converter.create_installation_guide()
    
    print("\n" + "=" * 50)
    print("✅ Conversion completed!")
    print(f"Converted {len(converted_models)} models:")
    
    for model_name, header_path in converted_models:
        print(f"  - {model_name}: {header_path.name}")
    
    print(f"\n📁 All files generated in: {converter.output_dir}")
    print("\n📋 Next steps:")
    print("1. Copy all .h files to your Arduino sketch folder")
    print("2. Install EloquentTinyML library in Arduino IDE")
    print("3. Upload esp32_eloquent_inference.ino to esp32")
    print("4. Monitor Serial output for results")
    
    return converter.output_dir

if __name__ == "__main__":
    output_dir = main()