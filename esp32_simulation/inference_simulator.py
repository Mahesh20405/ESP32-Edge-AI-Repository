import time
import psutil
import numpy as np
import tensorflow as tf
from tensorflow import keras
import logging
from pathlib import Path

class ESP32Simulator:
    """
    Simulates ESP32 inference performance including timing and memory usage
    """
    
    def __init__(self, model_path, target_memory_kb=32, target_latency_ms=50):
        self.model_path = model_path
        self.target_memory_kb = target_memory_kb
        self.target_latency_ms = target_latency_ms
        self.model = None
        self.inference_times = []
        self.memory_usage = []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def load_model(self):
        """Load TensorFlow Lite model for inference"""
        try:
            self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            
            # Get input and output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            self.logger.info(f"Model loaded successfully from {self.model_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False
    
    def simulate_inference(self, input_data, num_runs=100):
        """
        Simulate inference on ESP32-like constraints
        """
        if self.interpreter is None:
            self.logger.error("Model not loaded")
            return None
        
        results = {
            'inference_times': [],
            'memory_usage': [],
            'predictions': [],
            'accuracy_metrics': {}
        }
        
        for i in range(num_runs):
            # Measure memory before inference
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024  # KB
            
            # Simulate ESP32 memory constraints
            if memory_before > self.target_memory_kb * 1000:  # Convert to bytes
                self.logger.warning(f"Memory usage exceeds ESP32 limit: {memory_before/1024:.2f}MB")
            
            # Time inference
            start_time = time.time()
            
            # Set input tensor
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data[i:i+1])
            
            # Run inference
            self.interpreter.invoke()
            
            # Get output
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            end_time = time.time()
            
            # Calculate inference time in milliseconds
            inference_time = (end_time - start_time) * 1000
            
            # Measure memory after inference
            memory_after = process.memory_info().rss / 1024  # KB
            
            # Store results
            results['inference_times'].append(inference_time)
            results['memory_usage'].append(memory_after - memory_before)
            results['predictions'].append(output_data)
            
            # Check if within ESP32 constraints
            if inference_time > self.target_latency_ms:
                self.logger.warning(f"Inference time exceeds target: {inference_time:.2f}ms")
        
        # Calculate statistics
        results['avg_inference_time'] = np.mean(results['inference_times'])
        results['max_inference_time'] = np.max(results['inference_times'])
        results['min_inference_time'] = np.min(results['inference_times'])
        results['avg_memory_usage'] = np.mean(results['memory_usage'])
        
        return results
    
    def generate_performance_report(self, results, output_path="results/logs/performance_report.txt"):
        """Generate detailed performance report"""
        report = f"""
ESP32 Inference Performance Report
================================

Model: {self.model_path}
Target Latency: {self.target_latency_ms}ms
Target Memory: {self.target_memory_kb}KB

Performance Metrics:
- Average Inference Time: {results['avg_inference_time']:.2f}ms
- Maximum Inference Time: {results['max_inference_time']:.2f}ms
- Minimum Inference Time: {results['min_inference_time']:.2f}ms
- Average Memory Usage: {results['avg_memory_usage']:.2f}KB

Constraints Check:
- Latency Target Met: {'✓' if results['avg_inference_time'] <= self.target_latency_ms else '✗'}
- Memory Target Met: {'✓' if results['avg_memory_usage'] <= self.target_memory_kb else '✗'}

Recommendations:
"""
        
        if results['avg_inference_time'] > self.target_latency_ms:
            report += "- Consider further model optimization (pruning/quantization)\n"
        if results['avg_memory_usage'] > self.target_memory_kb:
            report += "- Reduce model size or use more aggressive compression\n"
        
        # Save report
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report)
        
        print(report)