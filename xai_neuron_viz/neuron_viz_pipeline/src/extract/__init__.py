# FILE: neuron_viz_pipeline/src/extract/__init__.py
# Package marker for neuron_viz_pipeline/src/extract.
#
# Exposes the ActivationExtractor class at package level:
#   from src.extract import ActivationExtractor

from .activations import ActivationExtractor

__all__ = ["ActivationExtractor"]