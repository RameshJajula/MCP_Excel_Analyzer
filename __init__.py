"""
MCP Excel Analyzer Package
Model Context Protocol server for Excel data analysis with local DeepSeek LLM
"""

__version__ = "1.0.0"
__author__ = "MCP Excel Analyzer Team"

from .mcp_server import ExcelAnalyzerMCPServer
from .llm_integration import DeepSeekAnalyzer, analyze_with_deepseek, generate_insights_with_deepseek

__all__ = [
    "ExcelAnalyzerMCPServer",
    "DeepSeekAnalyzer", 
    "analyze_with_deepseek",
    "generate_insights_with_deepseek"
] 