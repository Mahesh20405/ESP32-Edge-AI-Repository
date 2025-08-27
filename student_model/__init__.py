"""
Student Model Package for ESP32 Edge AI

This package contains lightweight student models optimized for ESP32 deployment
and knowledge distillation utilities for training with teacher guidance.
"""

from .train_student_kd import train_student_with_distillation, create_student_model
from .distillation_utils import (
    KnowledgeDistillationLoss,
    DistillationTrainer,
    temperature_scaling,
    compute_distillation_metrics
)

__version__ = "1.0.0"
__author__ = "ESP32 Edge AI Research Team"

__all__ = [
    'train_student_with_distillation',
    'create_student_model',
    'KnowledgeDistillationLoss', 
    'DistillationTrainer',
    'temperature_scaling',
    'compute_distillation_metrics'
]