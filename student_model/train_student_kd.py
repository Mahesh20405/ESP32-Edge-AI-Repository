"""
Student Model Training with Knowledge Distillation
Implements knowledge distillation to train a lightweight student model.
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import yaml
import argparse

class DistillationLoss(keras.losses.Loss):
    """Custom loss function for knowledge distillation"""
    
    def __init__(self, temperature=4.0, alpha=0.7, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature
        self.alpha = alpha
    
    def call(self, y_true, y_pred):
        # y_true contains both ground truth and teacher predictions
        student_pred = y_pred
        teacher_pred = y_true[..., 10:]  # Assuming 10 classes, teacher preds are last 10
        ground_truth = y_true[..., :10]  # Ground truth is first 10
        
        # Distillation loss
        teacher_soft = tf.nn.softmax(teacher_pred / self.temperature)
        student_soft = tf.nn.softmax(student_pred / self.temperature)
        distillation_loss = tf.keras.losses.categorical_crossentropy(
            teacher_soft, student_soft
        ) * (self.temperature ** 2)
        
        # Student loss
        student_loss = tf.keras.losses.categorical_crossentropy(
            ground_truth, tf.nn.softmax(student_pred)
        )
        
        return self.alpha * distillation_loss + (1 - self.alpha) * student_loss

def create_student_model(input_shape, num_classes=10):
    """
    Create a lightweight student model optimized for ESP32
    """
    model = keras.Sequential([
        keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

def train_student_with_kd(teacher_model, config):
    """
    Train student model using knowledge distillation
    """
    # Load data
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Preprocess data
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    y_train_cat = keras.utils.to_categorical(y_train, 10)
    y_test_cat = keras.utils.to_categorical(y_test, 10)
    
    # Get teacher predictions
    teacher_train_pred = teacher_model.predict(x_train, verbose=0)
    teacher_test_pred = teacher_model.predict(x_test, verbose=0)
    
    # Combine ground truth and teacher predictions
    y_train_combined = np.concatenate([y_train_cat, teacher_train_pred], axis=1)
    y_test_combined = np.concatenate([y_test_cat, teacher_test_pred], axis=1)
    
    # Create student model
    student_model = create_student_model((28, 28, 1), 10)
    
    # Compile with distillation loss
    student_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config['student']['learning_rate']),
        loss=DistillationLoss(
            temperature=config['kd']['temperature'],
            alpha=config['kd']['alpha']
        ),
        metrics=['accuracy']
    )
    
    # Train student
    history = student_model.fit(
        x_train, y_train_combined,
        batch_size=config['student']['batch_size'],
        epochs=config['student']['epochs'],
        validation_data=(x_test, y_test_combined),
        verbose=1
    )
    
    # Save student model
    student_model.save('results/models/student/student_model.h5')
    
    return student_model, history

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/hyperparameters.yaml')
    parser.add_argument('--teacher', default='results/models/teacher/teacher_model.h5')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    teacher_model = keras.models.load_model(args.teacher)
    student_model, history = train_student_with_kd(teacher_model, config)
    print("Student model training with KD completed!")