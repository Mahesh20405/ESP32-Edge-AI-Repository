# Distilling Intelligence: Deploying Lightweight Neural Networks on ESP32 for Edge AI

This repository contains the complete implementation of the research paper "Distilling Intelligence: Deploying Lightweight Neural Networks on ESP32 for Edge AI".

## Overview
This project demonstrates how to deploy lightweight neural networks on ESP32 microcontrollers using knowledge distillation, model compression techniques, and TensorFlow Lite Micro.

## Quick Start
1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the complete pipeline: `bash scripts/run_full_pipeline.sh`

## Project Structure
- `teacher_model/`: Training large teacher networks
- `student_model/`: Knowledge distillation implementation
- `optimization/`: Pruning and quantization scripts
- `tflite_conversion/`: Model conversion for deployment
- `esp32_simulation/`: Performance simulation scripts
- `datasets/`: Data loading and preprocessing
- `docs/`: Documentation and circuit diagrams

## Datasets
- **MNIST**: Handwritten digit classification
- **Synthetic Sensor Data**: Simulated IoT sensor readings

## Hyperparameters
| Parameter | Teacher Model | Student Model | KD Settings |
|-----------|---------------|---------------|-------------|
| Batch Size | 128 | 64 | 64 |
| Learning Rate | 0.001 | 0.0005 | 0.0005 |
| Epochs | 50 | 100 | 100 |
| KD Temperature | - | - | 4.0 |
| KD Alpha | - | - | 0.7 |

## Results
- Model Size Reduction: 95%
- Inference Time: <50ms on ESP32
- Memory Usage: <32KB RAM
- Accuracy Retention: >95%

## Hardware Setup
See [docs/HARDWARE_SETUP.md](docs/HARDWARE_SETUP.md) for ESP32 setup instructions and [docs/CIRCUIT_DIAGRAM.md](docs/CIRCUIT_DIAGRAM.md) for wiring diagrams.

## Citation
```bibtex
@inproceedings{esp32_edge_ai_2025,
  title={Distilling Intelligence: Deploying Lightweight Neural Networks on ESP32 for Edge AI},
  author={Preetha J, Mahesh R, Ajay Surya B, Thamizharasan P},
  booktitle={Conference Proceedings},
  year={2025}
}