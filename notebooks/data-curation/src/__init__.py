"""
Hooke Explain - Structured Biomedical Explanation Generation

This module provides tools for generating structured biomedical explanations
from reports using LLMs and managing enhanced chat interactions.
"""

from .llm_client import LLMClient, LLMConfig, create_llm_client
from .response_parser import ResponseParser, ParsedResponse
from .report_processor import ReportProcessor, ProcessedReport
from .structure_explainer import StructureExplainer, create_structure_explainer
from .enhanced_chat_manager import EnhancedChatManager, ChatConfiguration

__version__ = "0.1.0"
__all__ = [
    "LLMClient",
    "LLMConfig",
    "create_llm_client",
    "ResponseParser",
    "ParsedResponse",
    "ReportProcessor",
    "ProcessedReport",
    "StructureExplainer",
    "create_structure_explainer",
    "EnhancedChatManager",
    "ChatConfiguration"
] 