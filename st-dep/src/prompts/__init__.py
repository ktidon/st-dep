"""
Prompt templates for RAG chains.

This module contains:
- Technical documentation prompts
- Vision-based analysis prompts
- Semiconductor industry-specific templates
"""

from .technical_prompts import TECHNICAL_RAG_PROMPT
from .vision_prompts import VISION_RAG_PROMPT

__all__ = [
    'TECHNICAL_RAG_PROMPT',
    'VISION_RAG_PROMPT'
]