"""Explainable AI across three tasks and two modalities.

Public entry points:
    xai.config   -- configuration, seeding, device selection
    xai.data     -- dataset loading for all three tasks
    xai.models   -- unified model wrappers exposing ``predict_proba``
    xai.run      -- orchestration: explain every prediction, resumably
"""

__version__ = "2.0.0"
