"""
Knowledge Distillation Utilities for ESP32 Edge AI

This module provides utilities for knowledge distillation including custom loss functions,
training procedures, and evaluation metrics specifically designed for edge deployment.
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from typing import Tuple, Dict, Any, Optional, Callable
import logging
from pathlib import Path
import json


class KnowledgeDistillationLoss(keras.losses.Loss):
    """
    Custom loss function for knowledge distillation combining hard and soft targets
    """
    
    def __init__(self, alpha: float = 0.7, temperature: float = 4.0, 
                 reduction: str = 'auto', name: str = 'kd_loss'):
        """
        Initialize knowledge distillation loss
        
        Args:
            alpha: Weighting factor for distillation loss (0-1)
            temperature: Temperature for softmax scaling
            reduction: Type of reduction to apply
            name: Name of the loss function
        """
        super().__init__(reduction=reduction, name=name)
        self.alpha = alpha
        self.temperature = temperature
        
    def call(self, y_true, y_pred):
        """
        Compute knowledge distillation loss
        
        Args:
            y_true: Ground truth labels (one-hot encoded)
            y_pred: Student predictions
            
        Note: Teacher predictions should be passed via custom training step
        """
        # Standard cross-entropy loss (hard targets)
        hard_loss = keras.losses.categorical_crossentropy(y_true, y_pred)
        
        return hard_loss  # Soft loss computed in custom training step
    
    def distillation_loss(self, teacher_logits, student_logits, temperature):
        """
        Compute the distillation component of the loss
        
        Args:
            teacher_logits: Teacher model outputs (before softmax)
            student_logits: Student model outputs (before softmax)
            temperature: Temperature for scaling
            
        Returns:
            Distillation loss value
        """
        # Apply temperature scaling
        teacher_probs = tf.nn.softmax(teacher_logits / temperature)
        student_log_probs = tf.nn.log_softmax(student_logits / temperature)
        
        # KL divergence loss
        kl_loss = tf.reduce_sum(teacher_probs * 
                               (tf.math.log(teacher_probs + 1e-8) - student_log_probs), 
                               axis=1)
        
        return tf.reduce_mean(kl_loss) * (temperature ** 2)
    
    def get_config(self):
        """Return the config of the loss function"""
        config = super().get_config()
        config.update({
            'alpha': self.alpha,
            'temperature': self.temperature
        })
        return config


class DistillationTrainer:
    """
    Custom trainer for knowledge distillation with comprehensive logging and monitoring
    """
    
    def __init__(self, student_model: keras.Model, teacher_model: keras.Model,
                 alpha: float = 0.7, temperature: float = 4.0):
        """
        Initialize distillation trainer
        
        Args:
            student_model: Lightweight student model
            teacher_model: Pre-trained teacher model
            alpha: Balance between hard and soft targets
            temperature: Temperature for knowledge distillation
        """
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.alpha = alpha
        self.temperature = temperature
        
        # Freeze teacher model
        self.teacher_model.trainable = False
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Training history
        self.history = {
            'loss': [], 'hard_loss': [], 'soft_loss': [],
            'accuracy': [], 'val_loss': [], 'val_accuracy': [],
            'temperature_history': [], 'alpha_history': []
        }
    
    @tf.function
    def train_step(self, x_batch, y_batch):
        """
        Custom training step for knowledge distillation
        
        Args:
            x_batch: Input batch
            y_batch: Target batch (one-hot encoded)
            
        Returns:
            Dictionary of loss components and metrics
        """
        with tf.GradientTape() as tape:
            # Get predictions
            student_logits = self.student_model(x_batch, training=True)
            teacher_logits = self.teacher_model(x_batch, training=False)
            
            # Compute hard target loss (student vs ground truth)
            hard_loss = keras.losses.categorical_crossentropy(
                y_batch, tf.nn.softmax(student_logits)
            )
            hard_loss = tf.reduce_mean(hard_loss)
            
            # Compute soft target loss (student vs teacher)
            teacher_probs = tf.nn.softmax(teacher_logits / self.temperature)
            student_probs = tf.nn.softmax(student_logits / self.temperature)
            
            soft_loss = keras.losses.categorical_crossentropy(
                teacher_probs, student_probs
            )
            soft_loss = tf.reduce_mean(soft_loss) * (self.temperature ** 2)
            
            # Combined loss
            total_loss = (1 - self.alpha) * hard_loss + self.alpha * soft_loss
        
        # Compute gradients and update weights
        gradients = tape.gradient(total_loss, self.student_model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.student_model.trainable_variables)
        )
        
        # Calculate accuracy
        student_predictions = tf.nn.softmax(student_logits)
        accuracy = keras.metrics.categorical_accuracy(y_batch, student_predictions)
        accuracy = tf.reduce_mean(accuracy)
        
        return {
            'total_loss': total_loss,
            'hard_loss': hard_loss,
            'soft_loss': soft_loss,
            'accuracy': accuracy
        }
    
    def fit(self, train_dataset, validation_dataset=None, epochs: int = 100,
            optimizer=None, callbacks=None, verbose: int = 1):
        """
        Train student model with knowledge distillation
        
        Args:
            train_dataset: Training dataset
            validation_dataset: Validation dataset (optional)
            epochs: Number of training epochs
            optimizer: Optimizer instance
            callbacks: List of Keras callbacks
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        if optimizer is None:
            optimizer = keras.optimizers.Adam(learning_rate=0.001)
        
        self.optimizer = optimizer
        
        # Training loop
        for epoch in range(epochs):
            if verbose:
                print(f"Epoch {epoch + 1}/{epochs}")
            
            # Training phase
            epoch_losses = []
            epoch_hard_losses = []
            epoch_soft_losses = []
            epoch_accuracies = []
            
            for step, (x_batch, y_batch) in enumerate(train_dataset):
                metrics = self.train_step(x_batch, y_batch)
                
                epoch_losses.append(metrics['total_loss'])
                epoch_hard_losses.append(metrics['hard_loss'])
                epoch_soft_losses.append(metrics['soft_loss'])
                epoch_accuracies.append(metrics['accuracy'])
                
                if verbose and step % 100 == 0:
                    print(f"  Step {step}: loss={metrics['total_loss']:.4f}, "
                          f"accuracy={metrics['accuracy']:.4f}")
            
            # Calculate epoch metrics
            avg_loss = np.mean(epoch_losses)
            avg_hard_loss = np.mean(epoch_hard_losses)
            avg_soft_loss = np.mean(epoch_soft_losses)
            avg_accuracy = np.mean(epoch_accuracies)
            
            self.history['loss'].append(float(avg_loss))
            self.history['hard_loss'].append(float(avg_hard_loss))
            self.history['soft_loss'].append(float(avg_soft_loss))
            self.history['accuracy'].append(float(avg_accuracy))
            self.history['temperature_history'].append(self.temperature)
            self.history['alpha_history'].append(self.alpha)
            
            # Validation phase
            if validation_dataset is not None:
                val_metrics = self.evaluate(validation_dataset)
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_accuracy'].append(val_metrics['accuracy'])
                
                if verbose:
                    print(f"  Train: loss={avg_loss:.4f}, acc={avg_accuracy:.4f}")
                    print(f"  Val: loss={val_metrics['loss']:.4f}, "
                          f"acc={val_metrics['accuracy']:.4f}")
            
            # Execute callbacks
            if callbacks:
                for callback in callbacks:
                    callback.on_epoch_end(epoch, self.history)
        
        return self.history
    
    def evaluate(self, dataset) -> Dict[str, float]:
        """
        Evaluate student model on given dataset
        
        Args:
            dataset: Evaluation dataset
            
        Returns:
            Dictionary of evaluation metrics
        """
        total_loss = 0
        total_accuracy = 0
        num_batches = 0
        
        for x_batch, y_batch in dataset:
            student_logits = self.student_model(x_batch, training=False)
            teacher_logits = self.teacher_model(x_batch, training=False)
            
            # Compute losses
            hard_loss = keras.losses.categorical_crossentropy(
                y_batch, tf.nn.softmax(student_logits)
            )
            
            teacher_probs = tf.nn.softmax(teacher_logits / self.temperature)
            student_probs = tf.nn.softmax(student_logits / self.temperature)
            soft_loss = keras.losses.categorical_crossentropy(
                teacher_probs, student_probs
            ) * (self.temperature ** 2)
            
            batch_loss = (1 - self.alpha) * tf.reduce_mean(hard_loss) + \
                        self.alpha * tf.reduce_mean(soft_loss)
            
            # Compute accuracy
            student_predictions = tf.nn.softmax(student_logits)
            batch_accuracy = keras.metrics.categorical_accuracy(
                y_batch, student_predictions
            )
            
            total_loss += batch_loss
            total_accuracy += tf.reduce_mean(batch_accuracy)
            num_batches += 1
        
        return {
            'loss': float(total_loss / num_batches),
            'accuracy': float(total_accuracy / num_batches)
        }
    
    def save_history(self, filepath: str):
        """Save training history to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)


def temperature_scaling(logits: tf.Tensor, temperature: float) -> tf.Tensor:
    """
    Apply temperature scaling to logits
    
    Args:
        logits: Raw model outputs
        temperature: Scaling temperature
        
    Returns:
        Temperature-scaled probabilities
    """
    return tf.nn.softmax(logits / temperature)


def compute_distillation_metrics(teacher_model: keras.Model, 
                                student_model: keras.Model,
                                test_dataset, 
                                temperature: float = 4.0) -> Dict[str, float]:
    """
    Compute comprehensive metrics for knowledge distillation evaluation
    
    Args:
        teacher_model: Teacher model
        student_model: Student model  
        test_dataset: Test dataset
        temperature: Temperature for soft targets
        
    Returns:
        Dictionary of evaluation metrics
    """
    teacher_correct = 0
    student_correct = 0
    agreement = 0
    kl_divergence_sum = 0
    total_samples = 0
    
    for x_batch, y_batch in test_dataset:
        # Get predictions
        teacher_logits = teacher_model(x_batch, training=False)
        student_logits = student_model(x_batch, training=False)
        
        teacher_probs = tf.nn.softmax(teacher_logits)
        student_probs = tf.nn.softmax(student_logits)
        
        # Compute accuracies
        teacher_pred = tf.argmax(teacher_probs, axis=1)
        student_pred = tf.argmax(student_probs, axis=1)
        true_labels = tf.argmax(y_batch, axis=1)
        
        teacher_correct += tf.reduce_sum(tf.cast(
            tf.equal(teacher_pred, true_labels), tf.int32
        ))
        student_correct += tf.reduce_sum(tf.cast(
            tf.equal(student_pred, true_labels), tf.int32
        ))
        
        # Teacher-student agreement
        agreement += tf.reduce_sum(tf.cast(
            tf.equal(teacher_pred, student_pred), tf.int32
        ))
        
        # KL divergence between teacher and student
        teacher_soft = temperature_scaling(teacher_logits, temperature)
        student_soft = temperature_scaling(student_logits, temperature)
        
        kl_div = tf.reduce_sum(teacher_soft * tf.math.log(
            teacher_soft / (student_soft + 1e-8) + 1e-8
        ), axis=1)
        kl_divergence_sum += tf.reduce_sum(kl_div)
        
        total_samples += x_batch.shape[0]
    
    # Calculate final metrics
    teacher_accuracy = float(teacher_correct / total_samples)
    student_accuracy = float(student_correct / total_samples)
    agreement_rate = float(agreement / total_samples)
    avg_kl_divergence = float(kl_divergence_sum / total_samples)
    
    # Knowledge retention rate
    knowledge_retention = student_accuracy / teacher_accuracy if teacher_accuracy > 0 else 0
    
    return {
        'teacher_accuracy': teacher_accuracy,
        'student_accuracy': student_accuracy, 
        'agreement_rate': agreement_rate,
        'avg_kl_divergence': avg_kl_divergence,
        'knowledge_retention': knowledge_retention,
        'accuracy_drop': teacher_accuracy - student_accuracy
    }


class AdaptiveTemperatureScheduler:
    """
    Scheduler for adaptive temperature during knowledge distillation training
    """
    
    def __init__(self, initial_temperature: float = 4.0, 
                 decay_rate: float = 0.95, min_temperature: float = 1.0):
        self.initial_temperature = initial_temperature
        self.current_temperature = initial_temperature
        self.decay_rate = decay_rate
        self.min_temperature = min_temperature
    
    def update(self, epoch: int, validation_loss: float = None):
        """Update temperature based on training progress"""
        if validation_loss is not None and hasattr(self, 'prev_loss'):
            # Adapt based on validation loss improvement
            if validation_loss < self.prev_loss:
                # Loss improved, reduce temperature
                self.current_temperature = max(
                    self.current_temperature * self.decay_rate,
                    self.min_temperature
                )
        else:
            # Simple decay schedule
            self.current_temperature = max(
                self.initial_temperature * (self.decay_rate ** epoch),
                self.min_temperature
            )
        
        if validation_loss is not None:
            self.prev_loss = validation_loss
        
        return self.current_temperature


# Example usage and testing
if __name__ == "__main__":
    # Test knowledge distillation utilities
    print("Testing Knowledge Distillation Utilities")
    print("=" * 50)
    
    # Create dummy models for testing
    input_shape = (28, 28, 1)
    num_classes = 10
    
    # Simple teacher model
    teacher = keras.Sequential([
        keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    # Simple student model
    student = keras.Sequential([
        keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Flatten(),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    print(f"Teacher parameters: {teacher.count_params():,}")
    print(f"Student parameters: {student.count_params():,}")
    print(f"Compression ratio: {teacher.count_params() / student.count_params():.2f}x")
    
    # Test distillation loss
    kd_loss = KnowledgeDistillationLoss(alpha=0.7, temperature=4.0)
    print(f"KD Loss initialized: alpha={kd_loss.alpha}, T={kd_loss.temperature}")
    
    # Test temperature scheduler
    temp_scheduler = AdaptiveTemperatureScheduler()
    print("Temperature schedule:")
    for epoch in range(10):
        temp = temp_scheduler.update(epoch)
        print(f"  Epoch {epoch}: T = {temp:.3f}")