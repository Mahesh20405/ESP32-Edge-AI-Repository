"""
Teacher Model Architectures for Knowledge Distillation

This module defines various teacher model architectures optimized for
generating rich feature representations for knowledge transfer.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from typing import Tuple, Optional, Dict, Any
import numpy as np


class TeacherCNN:
    """
    Convolutional Neural Network teacher model with high capacity
    """
    
    def __init__(self, input_shape: Tuple[int, int, int], num_classes: int = 10):
        self.input_shape = input_shape
        self.num_classes = num_classes
    
    def build_model(self, dropout_rate: float = 0.5) -> keras.Model:
        """
        Build a high-capacity CNN teacher model
        
        Args:
            dropout_rate: Dropout rate for regularization
            
        Returns:
            Compiled Keras model
        """
        model = models.Sequential([
            # First convolutional block
            layers.Conv2D(64, (3, 3), activation='relu', input_shape=self.input_shape,
                         padding='same', name='conv1_1'),
            layers.BatchNormalization(name='bn1_1'),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2'),
            layers.BatchNormalization(name='bn1_2'),
            layers.MaxPooling2D((2, 2), name='pool1'),
            layers.Dropout(dropout_rate/2, name='dropout1'),
            
            # Second convolutional block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1'),
            layers.BatchNormalization(name='bn2_1'),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2'),
            layers.BatchNormalization(name='bn2_2'),
            layers.MaxPooling2D((2, 2), name='pool2'),
            layers.Dropout(dropout_rate/2, name='dropout2'),
            
            # Third convolutional block
            layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1'),
            layers.BatchNormalization(name='bn3_1'),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2'),
            layers.BatchNormalization(name='bn3_2'),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_3'),
            layers.BatchNormalization(name='bn3_3'),
            layers.MaxPooling2D((2, 2), name='pool3'),
            layers.Dropout(dropout_rate, name='dropout3'),
            
            # Dense layers
            layers.Flatten(name='flatten'),
            layers.Dense(1024, activation='relu', name='dense1'),
            layers.BatchNormalization(name='bn_dense1'),
            layers.Dropout(dropout_rate, name='dropout_dense1'),
            layers.Dense(512, activation='relu', name='dense2'),
            layers.BatchNormalization(name='bn_dense2'),
            layers.Dropout(dropout_rate, name='dropout_dense2'),
            layers.Dense(self.num_classes, activation='softmax', name='predictions')
        ])
        
        return model


class TeacherResNet:
    """
    ResNet-based teacher model with residual connections
    """
    
    def __init__(self, input_shape: Tuple[int, int, int], num_classes: int = 10):
        self.input_shape = input_shape
        self.num_classes = num_classes
    
    def residual_block(self, x, filters: int, stride: int = 1, name: str = ""):
        """Create a residual block"""
        shortcut = x
        
        # First conv layer
        x = layers.Conv2D(filters, (3, 3), strides=stride, padding='same',
                         name=f'{name}_conv1')(x)
        x = layers.BatchNormalization(name=f'{name}_bn1')(x)
        x = layers.ReLU(name=f'{name}_relu1')(x)
        
        # Second conv layer
        x = layers.Conv2D(filters, (3, 3), padding='same', name=f'{name}_conv2')(x)
        x = layers.BatchNormalization(name=f'{name}_bn2')(x)
        
        # Shortcut connection
        if stride != 1 or shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, (1, 1), strides=stride,
                                   name=f'{name}_shortcut_conv')(shortcut)
            shortcut = layers.BatchNormalization(name=f'{name}_shortcut_bn')(shortcut)
        
        x = layers.Add(name=f'{name}_add')([x, shortcut])
        x = layers.ReLU(name=f'{name}_relu2')(x)
        
        return x
    
    def build_model(self) -> keras.Model:
        """Build ResNet teacher model"""
        inputs = layers.Input(shape=self.input_shape)
        
        # Initial convolution
        x = layers.Conv2D(64, (7, 7), strides=2, padding='same', name='conv1')(inputs)
        x = layers.BatchNormalization(name='bn1')(x)
        x = layers.ReLU(name='relu1')(x)
        x = layers.MaxPooling2D((3, 3), strides=2, padding='same', name='pool1')(x)
        
        # Residual blocks
        x = self.residual_block(x, 64, name='res2a')
        x = self.residual_block(x, 64, name='res2b')
        
        x = self.residual_block(x, 128, stride=2, name='res3a')
        x = self.residual_block(x, 128, name='res3b')
        
        x = self.residual_block(x, 256, stride=2, name='res4a')
        x = self.residual_block(x, 256, name='res4b')
        
        x = self.residual_block(x, 512, stride=2, name='res5a')
        x = self.residual_block(x, 512, name='res5b')
        
        # Global average pooling and classification
        x = layers.GlobalAveragePooling2D(name='global_pool')(x)
        x = layers.Dense(self.num_classes, activation='softmax', name='predictions')(x)
        
        return models.Model(inputs, x, name='teacher_resnet')


class TeacherMobileNet:
    """
    MobileNet-based teacher model (larger version)
    """
    
    def __init__(self, input_shape: Tuple[int, int, int], num_classes: int = 10):
        self.input_shape = input_shape
        self.num_classes = num_classes
    
    def build_model(self, alpha: float = 1.4) -> keras.Model:
        """
        Build MobileNet teacher model
        
        Args:
            alpha: Width multiplier for making the model larger
            
        Returns:
            Compiled Keras model
        """
        base_model = keras.applications.MobileNetV2(
            input_shape=self.input_shape,
            alpha=alpha,
            include_top=False,
            weights=None  # Train from scratch
        )
        
        # Add custom classification head
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model


def get_teacher_model(architecture: str, input_shape: Tuple[int, int, int], 
                     num_classes: int = 10, **kwargs) -> keras.Model:
    """
    Factory function to create teacher models
    
    Args:
        architecture: Type of architecture ('cnn', 'resnet', 'mobilenet')
        input_shape: Input tensor shape
        num_classes: Number of output classes
        **kwargs: Additional architecture-specific parameters
        
    Returns:
        Keras model instance
        
    Raises:
        ValueError: If architecture is not supported
    """
    architecture = architecture.lower()
    
    if architecture == 'cnn':
        teacher = TeacherCNN(input_shape, num_classes)
        return teacher.build_model(**kwargs)
    
    elif architecture == 'resnet':
        teacher = TeacherResNet(input_shape, num_classes)
        return teacher.build_model(**kwargs)
    
    elif architecture == 'mobilenet':
        teacher = TeacherMobileNet(input_shape, num_classes)
        return teacher.build_model(**kwargs)
    
    else:
        raise ValueError(f"Unsupported architecture: {architecture}. "
                        f"Choose from: 'cnn', 'resnet', 'mobilenet'")


def get_model_complexity_info(model: keras.Model) -> Dict[str, Any]:
    """
    Calculate model complexity metrics
    
    Args:
        model: Keras model
        
    Returns:
        Dictionary with complexity metrics
    """
    total_params = model.count_params()
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params
    
    # Estimate model size in MB (assuming float32)
    model_size_mb = (total_params * 4) / (1024 * 1024)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': non_trainable_params,
        'model_size_mb': round(model_size_mb, 3),
        'layers': len(model.layers)
    }


# Example usage and testing
if __name__ == "__main__":
    # Test different teacher architectures
    input_shape = (28, 28, 1)  # MNIST
    num_classes = 10
    
    print("Testing Teacher Model Architectures")
    print("=" * 50)
    
    for arch in ['cnn', 'resnet', 'mobilenet']:
        print(f"\nTesting {arch.upper()} architecture:")
        model = get_teacher_model(arch, input_shape, num_classes)
        complexity = get_model_complexity_info(model)
        
        print(f"  Parameters: {complexity['total_parameters']:,}")
        print(f"  Model size: {complexity['model_size_mb']:.3f} MB")
        print(f"  Layers: {complexity['layers']}")
        
        # Test forward pass
        dummy_input = tf.random.normal((1,) + input_shape)
        output = model(dummy_input, training=False)
        print(f"  Output shape: {output.shape}")