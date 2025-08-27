import tensorflow as tf
import tensorflow_model_optimization as tfmot
import numpy as np
from pathlib import Path

class ModelOptimizer:
    """
    Complete model optimization pipeline including pruning and quantization
    """
    
    def __init__(self, model_path):
        self.original_model = tf.keras.models.load_model(model_path)
        self.pruned_model = None
        self.quantized_model = None
    
    def structured_pruning(self, sparsity=0.8):
        """Apply structured pruning to reduce model size"""
        
        # Define pruning schedule
        pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=sparsity,
            begin_step=0,
            end_step=1000
        )
        
        # Apply pruning to dense layers only
        def apply_pruning_to_dense(layer):
            if isinstance(layer, tf.keras.layers.Dense):
                return tfmot.sparsity.keras.prune_low_magnitude(
                    layer, pruning_schedule=pruning_schedule
                )
            return layer
        
        # Clone and modify model
        self.pruned_model = tf.keras.models.clone_model(
            self.original_model,
            clone_function=apply_pruning_to_dense
        )
        
        return self.pruned_model
    
    def post_training_quantization(self, representative_data):
        """Apply post-training quantization"""
        
        # Convert to TensorFlow Lite with quantization
        converter = tf.lite.TFLiteConverter.from_keras_model(
            self.pruned_model if self.pruned_model else self.original_model
        )
        
        # Enable optimization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Set representative dataset for full integer quantization
        def representative_dataset():
            for data in representative_data:
                yield [data.astype(np.float32)]
        
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        
        # Convert model
        quantized_tflite_model = converter.convert()
        
        return quantized_tflite_model
    
    def get_model_size(self, model_path):
        """Get model size in bytes"""
        return Path(model_path).stat().st_size
    
    def compare_models(self, original_path, optimized_path):
        """Compare original and optimized model sizes"""
        original_size = self.get_model_size(original_path)
        optimized_size = self.get_model_size(optimized_path)
        
        size_reduction = (1 - optimized_size / original_size) * 100
        
        print(f"Original Model Size: {original_size / 1024:.2f} KB")
        print(f"Optimized Model Size: {optimized_size / 1024:.2f} KB")
        print(f"Size Reduction: {size_reduction:.1f}%")
        
        return size_reduction
