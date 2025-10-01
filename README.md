# Distilling Intelligence: Deploying Lightweight Neural Networks on ESP32 for Edge AI

This repository contains the complete implementation of the research paper "Distilling Intelligence: Deploying Lightweight Neural Networks on ESP32 for Edge AI".

## Overview
This project demonstrates how to deploy lightweight neural networks on ESP32 microcontrollers using knowledge distillation, model compression techniques, and TensorFlow Lite Micro. The implementation focuses on achieving real-time inference with minimal power consumption on resource-constrained edge devices.

## Hardware Specifications

### ESP32-WROOM-32D Target Platform
| Component | Specification |
|-----------|---------------|
| **Processor** | Dual-core Xtensa LX6 @ 240MHz |
| **Memory** | 520KB SRAM, 448KB ROM |
| **Flash Storage** | 4MB external SPI flash |
| **Wireless** | Wi-Fi 802.11 b/g/n, Bluetooth v4.2 |
| **Operating Voltage** | 3.0V - 3.6V |
| **Temperature Range** | -40°C to +85°C |
| **Power Consumption** | Active: 160-260mA, Deep Sleep: 10μA |

### Model Deployment Stack
| Component | Version/Configuration |
|-----------|----------------------|
| **TensorFlow Lite** | TensorFlow Lite for Microcontrollers v2.12.0 |
| **Development Framework** | ESP-IDF v4.4.4 |
| **Operating System** | FreeRTOS real-time OS |
| **Model Format** | TensorFlow Lite (.tflite) |
| **Quantization** | INT8 post-training quantization |
| **Memory Allocation** | 64KB heap reserved for inference |

## Quick Start
1. Clone this repository
   ```bash
   git clone https://github.com/your-repo/esp32-edge-ai.git
   cd esp32-edge-ai
   ```

2. Install Python dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Set up ESP-IDF development environment
   ```bash
   # Follow ESP-IDF installation guide for your platform
   # Ensure ESP-IDF v4.4.4 is installed
   ```

4. Run the complete training and optimization pipeline
   ```bash
   bash scripts/run_full_pipeline.sh
   ```

5. Deploy to ESP32
   ```bash
   bash scripts/deploy_to_esp32.sh
   ```

## Project Structure
```
├── teacher_model/          # Training scripts and configs for teacher network
│   ├── train.py
│   ├── model.py
│   └── ...
├── student_model/          # Student model, distillation scripts, configs
│   ├── distill.py
│   ├── model.py
│   └── ...
├── optimization/           # Pruning, quantization, and compression scripts
│   ├── prune.py
│   ├── quantize.py
│   └── ...
├── tflite_conversion/      # Scripts for TFLite model export and validation
│   ├── convert.py
│   └── ...
├── esp32_simulation/       # PC-side simulation and benchmarking tools
│   ├── simulate.py
│   └── ...
├── esp32_firmware/         # ESP32 C++ firmware (main, model loader, inference)
│   ├── main/
│   ├── components/
│   └── CMakeLists.txt
├── datasets/               # Data loaders, preprocessors, and sample data
│   ├── mnist/
│   ├── sensors/
│   └── ...
├── scripts/                # End-to-end automation and deployment scripts
│   ├── run_full_pipeline.sh
│   ├── deploy_to_esp32.sh
│   └── ...
├── docs/                   # Documentation, hardware setup, diagrams
│   ├── HARDWARE_SETUP.md
│   ├── CIRCUIT_DIAGRAM.md
│   └── ...
├── results/                # Experimental logs, metrics, plots
│   ├── mnist/
│   ├── sensors/
│   └── ...
├── requirements.txt        # Python dependencies
└── README.md               # Project overview and instructions
```

## Datasets

### MNIST Handwritten Digits
- **Size**: 70,000 images (60,000 train, 10,000 test)
- **Format**: 28×28 grayscale images
- **Classes**: 10 digit classes (0-9)
- **Use Case**: Image classification benchmark

### Synthetic Sensor Data
- **Size**: 100,000 samples
- **Features**: Temperature, humidity, pressure, accelerometer data
- **Format**: Time-series sequences
- **Use Case**: IoT sensor data classification

## Model Architecture & Hyperparameters

### Teacher Model (Desktop Training)
| Parameter | Value | Description |
|-----------|--------|-------------|
| **Architecture** | CNN (3 Conv + 2 FC layers) | Large capacity model |
| **Parameters** | ~1.2M | Full precision weights |
| **Batch Size** | 128 | Training batch size |
| **Learning Rate** | 0.001 | Adam optimizer |
| **Epochs** | 50 | Training iterations |
| **Input Size** | 28×28×1 (MNIST) | Grayscale images |

### Student Model (Edge Deployment)  
| Parameter | Value | Description |
|-----------|--------|-------------|
| **Architecture** | Lightweight CNN (2 Conv + 1 FC) | Optimized for ESP32 |
| **Parameters** | ~60K | Compressed model |
| **Batch Size** | 64 | Reduced for memory |
| **Learning Rate** | 0.0005 | Lower learning rate |
| **Epochs** | 100 | Extended training |

### Knowledge Distillation Settings
| Parameter | Value | Purpose |
|-----------|--------|---------|
| **Temperature** | 4.0 | Softmax temperature for soft targets |
| **Alpha** | 0.7 | Weight for distillation loss |
| **Beta** | 0.3 | Weight for ground truth loss |
| **Batch Size** | 64 | Memory-efficient training |

## Performance Results

### Model Compression Metrics
| Metric | Teacher | Student | Improvement |
|--------|---------|---------|-------------|
| **Model Size** | 4.8MB | 240KB | **95% reduction** |
| **Parameters** | 1.2M | 60K | **95% reduction** |
| **Accuracy (MNIST)** | 99.2% | 98.7% | **-0.5% loss** |
| **Accuracy (Sensor)** | 94.1% | 93.8% | **-0.3% loss** |

### ESP32 Runtime Performance
| Metric | Value | Target |
|--------|--------|--------|
| **Inference Time** | 45ms | <50ms ✓ |
| **Memory Usage (RAM)** | 28KB | <32KB ✓ |
| **Memory Usage (Flash)** | 180KB | <256KB ✓ |
| **Power Consumption** | 185mA @ 3.3V | <200mA ✓ |
| **CPU Utilization** | 73% @ 240MHz | <80% ✓ |

### Energy Efficiency
| Operation Mode | Current Draw | Battery Life (2000mAh) |
|----------------|--------------|----------------------|
| **Active Inference** | 185mA | ~10.8 hours |
| **Idle (Wi-Fi on)** | 80mA | ~25 hours |
| **Deep Sleep** | 10μA | ~22.8 years |

## Hardware Setup

### Required Components
- ESP32-WROOM-32D development board
- USB-C cable for programming
- Breadboard and jumper wires (for sensor connections)
- Optional: External sensors (temperature, accelerometer)

### Pin Configuration
```cpp
// GPIO pin assignments
#define LED_PIN     2    // Built-in LED
#define BUTTON_PIN  0    // Boot button
#define SDA_PIN    21    // I2C data (for sensors)
#define SCL_PIN    22    // I2C clock (for sensors)
```

For detailed hardware setup instructions, see [docs/HARDWARE_SETUP.md](docs/HARDWARE_SETUP.md)

For wiring diagrams and schematics, see [docs/CIRCUIT_DIAGRAM.md](docs/CIRCUIT_DIAGRAM.md)

## Software Architecture

### Training Pipeline
1. **Teacher Training**: Train large, accurate model on desktop
2. **Knowledge Distillation**: Transfer knowledge to compact student model  
3. **Post-Training Quantization**: Convert to INT8 precision
4. **Model Pruning**: Remove redundant connections
5. **TFLite Conversion**: Generate .tflite file for deployment

### ESP32 Firmware Structure
```cpp
// Main inference loop
void app_main() {
    // Initialize TensorFlow Lite Micro
    tflite::MicroInterpreter* interpreter = setup_model();
    
    while (true) {
        // Acquire sensor data or image
        float* input_data = get_input_data();
        
        // Run inference
        run_inference(interpreter, input_data);
        
        // Process results
        handle_prediction_results();
        
        // Power management
        vTaskDelay(pdMS_TO_TICKS(100));
    }
}
```

## Installation & Development

### Prerequisites
- Python 3.8+ with TensorFlow 2.12+
- ESP-IDF v4.4.4 development framework
- Git and CMake build tools

### Environment Setup
```bash
# Install ESP-IDF
mkdir -p ~/esp
cd ~/esp
git clone -b v4.4.4 --recursive https://github.com/espressif/esp-idf.git
cd esp-idf
./install.sh
. ./export.sh

# Install Python dependencies
pip install tensorflow==2.12.0 numpy matplotlib scikit-learn
```

### Building and Flashing
```bash
# Configure target chip
idf.py set-target esp32

# Build firmware
idf.py build

# Flash to device
idf.py -p /dev/ttyUSB0 flash monitor
```

## Experimental Results

### Knowledge Distillation Effectiveness
- Student model achieves 98.7% accuracy vs teacher's 99.2%
- 95% parameter reduction with only 0.5% accuracy loss
- Distillation temperature of 4.0 provides optimal knowledge transfer

### Quantization Impact
- INT8 quantization reduces model size by 75%
- Inference speedup of 2.1x on ESP32
- Negligible accuracy degradation (<0.2%)

### Real-world Performance
- Consistent <50ms inference across different input sizes
- Stable operation in -20°C to +60°C range
- 99.8% uptime over 7-day continuous operation test

## Troubleshooting

### Common Issues
1. **Memory Allocation Errors**: Reduce model size or increase heap allocation
2. **Slow Inference**: Enable compiler optimizations (`-O2`)  
3. **Wi-Fi Connectivity**: Check power supply stability
4. **Model Accuracy Drop**: Verify quantization parameters

### Debug Commands
```bash
# Monitor serial output
idf.py monitor

# Check memory usage
idf.py size

# Enable debug logging
idf.py menuconfig → Component config → Log output
```

## Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add improvement'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request


## Citation
```bibtex
@inproceedings{esp32_edge_ai_2025,
  title={Distilling Intelligence: Deploying Lightweight Neural Networks on ESP32 for Edge AI},
  author={Preetha J, Mahesh R, Ajay Surya B, Thamizharasan P},
  booktitle={IEEE International Conference on Edge Computing and AI},
  pages={1--8},
  year={2025},
  organization={IEEE}
}
```