"""
Teacher Model Package for ESP32 Edge AI

This package contains implementations for training large teacher models
that will be used for knowledge distillation to create lightweight student models.
"""

from .train_teacher import train_teacher_model, create_teacher_model
from .model_architectures import (
    TeacherCNN, 
    TeacherResNet, 
    TeacherMobileNet,
    get_teacher_model
)

__version__ = "1.0.0"
__author__ = "ESP32 Edge AI Research Team"

__all__ = [
    'train_teacher_model',
    'create_teacher_model', 
    'TeacherCNN',
    'TeacherResNet',
    'TeacherMobileNet',
    'get_teacher_model'
]