"""
Causal Query Generation (CQG) Pipeline

A production-ready pipeline for generating high-quality causal queries
from contact center transcripts.
"""

__version__ = "1.0.0"

from .transcript_loader import TranscriptLoader
from .action_extractor import ActionDrivenExtractor
from .cluster_query_generator import ClusterQueryGenerator
from .pipeline_runner import CQGPipeline
from .llm_client import LLMClient
from .config import Config

__all__ = [
    "TranscriptLoader",
    "ActionDrivenExtractor",
    "ClusterQueryGenerator",
    "CQGPipeline",
    "LLMClient",
    "Config",
]
