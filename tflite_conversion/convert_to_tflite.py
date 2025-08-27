import tensorflow as tf
import numpy as np
from pathlib import Path

def convert_to_tflite_micro(model_path, output_path, representative_data=None):
    """
    Convert Keras model to TensorFlow Lite Micro format
    """
    
    # Load the model
    model = tf.keras.models.load_model(model_path)
    
    # Create converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Set optimization flags for microcontroller deployment
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    if representative_data is not None:
        def representative_dataset():
            for data in representative_data:
                yield [data.astype(np.float32)]
        
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
    
    # Convert the model
    tflite_model = converter.convert()
    
    # Save the model
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"TensorFlow Lite model saved to {output_path}")
    
    return tflite_model

def generate_c_array(tflite_model_path, output_header_path):
    """
    Generate C header file with model data for ESP32 deployment
    """
    
    # Read the TFLite model
    with open(tflite_model_path, 'rb') as f:
        tflite_model = f.read()
    
    # Generate C array
    model_size = len(tflite_model)
    
    header_content = f"""
// Auto-generated header file for ESP32 deployment
// Model: {tflite_model_path}
// Size: {model_size} bytes

#ifndef MODEL_DATA_H
#define MODEL_DATA_H

const unsigned char model_data[] = {{
"""
    
    # Add model data as hex bytes
    for i, byte in enumerate(tflite_model):
        if i % 16 == 0:
            header_content += "\n  "
        header_content += f"0x{byte:02x}, "
    
    header_content += f"""
}};

const int model_data_len = {model_size};

#endif // MODEL_DATA_H
"""
    
    # Save header file
    Path(output_header_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_header_path, 'w') as f:
        f.write(header_content)
    
    print(f"C header file generated: {output_header_path}")
