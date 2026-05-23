# FILE: neuron_viz_pipeline/src/models/__init__.py
# Package marker for neuron_viz_pipeline/src/models.
#
# Exposes the model builder at package level:
#   from src.models import build_model

from .builder import build_model

__all__ = ["build_model"]