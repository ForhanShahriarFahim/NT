# FILE: neuron_viz_pipeline/src/xai/__init__.py
# Package marker for neuron_viz_pipeline/src/xai.
#
# Exposes the XAI method API at package level:
#   from src.xai import XAIMethod, GradientBasedMethod, IxG
#   from src.xai import get_xai_method, available_methods

from .base import XAIMethod, GradientBasedMethod
from .ixg import IxG
from .ig  import IntegratedGradients
from .attention_rollout import AttentionRollout
from .registry import get_xai_method, available_methods, METHODS

__all__ = [
    "XAIMethod",
    "GradientBasedMethod",
    "IxG",
    "IntegratedGradients",
    "AttentionRollout",
    "get_xai_method",
    "available_methods",
    "METHODS",
]