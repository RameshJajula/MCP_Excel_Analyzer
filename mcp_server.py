#!/usr/bin/env python3
"""
MCP Server for Excel Data Analysis with Local DeepSeek LLM
This server provides tools for reading, analyzing, and visualizing Excel data
using a local DeepSeek language model.
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequest,
    CallToolResult,
    ListToolsRequest,
    ListToolsResult,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
current_excel_file: Optional[str] = None
current_dataframe: Optional[pd.DataFrame] = None
llm_model = None

class ExcelAnalyzerMCPServer:
    def __init__(self):
        self.server = Server("excel-analyzer")
        self.setup_tools()
        
    def setup_tools(self):
        """Register all available tools with the MCP server."""
        
        @self.server.list_tools()
        async def handle_list_tools() -> ListToolsResult:
            """List all available tools."""
            return ListToolsResult(
                tools=[
                    Tool(
                        name="load_excel_file",
                        description="Load an Excel file for analysis",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "file_path": {
                                    "type": "string",
                                    "description": "Path to the Excel file"
                                }
                            },
                            "required": ["file_path"]
                        }
                    ),
                    Tool(
                        name="get_excel_info",
                        description="Get basic information about the loaded Excel file",
                        inputSchema={
                            "type": "object",
                            "properties": {}
                        }
                    ),
                    Tool(
                        name="analyze_data",
                        description="Perform comprehensive data analysis using local DeepSeek LLM",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "analysis_type": {
                                    "type": "string",
                                    "description": "Type of analysis: 'basic', 'statistical', 'correlation', 'trends', 'outliers', 'comprehensive'",
                                    "enum": ["basic", "statistical", "correlation", "trends", "outliers", "comprehensive"]
                                },
                                "columns": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Specific columns to analyze (optional)"
                                }
                            }
                        }
                    ),
                    Tool(
                        name="create_visualization",
                        description="Create data visualizations",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "chart_type": {
                                    "type": "string",
                                    "description": "Type of chart: 'histogram', 'scatter', 'line', 'bar', 'correlation', 'boxplot'",
                                    "enum": ["histogram", "scatter", "line", "bar", "correlation", "boxplot"]
                                },
                                "x_column": {
                                    "type": "string",
                                    "description": "Column for x-axis"
                                },
                                "y_column": {
                                    "type": "string",
                                    "description": "Column for y-axis"
                                },
                                "title": {
                                    "type": "string",
                                    "description": "Chart title"
                                }
                            },
                            "required": ["chart_type"]
                        }
                    ),
                    Tool(
                        name="generate_insights",
                        description="Generate business insights and recommendations using DeepSeek LLM",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "focus_area": {
                                    "type": "string",
                                    "description": "Specific area to focus on (e.g., 'sales', 'performance', 'trends')"
                                },
                                "business_context": {
                                    "type": "string",
                                    "description": "Additional business context for better insights"
                                }
                            }
                        }
                    ),
                    Tool(
                        name="export_analysis",
                        description="Export analysis results to various formats",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "format": {
                                    "type": "string",
                                    "description": "Export format: 'excel', 'csv', 'json', 'html'",
                                    "enum": ["excel", "csv", "json", "html"]
                                },
                                "output_path": {
                                    "type": "string",
                                    "description": "Output file path"
                                }
                            },
                            "required": ["format"]
                        }
                    )
                ]
            )

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> CallToolResult:
            """Handle tool calls."""
            try:
                if name == "load_excel_file":
                    return await self.load_excel_file(arguments)
                elif name == "get_excel_info":
                    return await self.get_excel_info()
                elif name == "analyze_data":
                    return await self.analyze_data(arguments)
                elif name == "create_visualization":
                    return await self.create_visualization(arguments)
                elif name == "generate_insights":
                    return await self.generate_insights(arguments)
                elif name == "export_analysis":
                    return await self.export_analysis(arguments)
                else:
                    raise ValueError(f"Unknown tool: {name}")
            except Exception as e:
                logger.error(f"Error in tool {name}: {str(e)}")
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Error: {str(e)}")]
                )

    async def load_excel_file(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Load an Excel file into memory."""
        global current_excel_file, current_dataframe
        
        file_path = arguments["file_path"]
        
        if not os.path.exists(file_path):
            return CallToolResult(
                content=[TextContent(type="text", text=f"Error: File {file_path} not found")]
            )
        
        try:
            # Read Excel file
            excel_file = pd.ExcelFile(file_path)
            sheet_names = excel_file.sheet_names
            
            # Load first sheet by default
            current_dataframe = pd.read_excel(file_path, sheet_name=sheet_names[0])
            current_excel_file = file_path
            
            info = f"""
✅ Excel file loaded successfully!
📁 File: {file_path}
📊 Sheets: {', '.join(sheet_names)}
📋 Current sheet: {sheet_names[0]}
📈 Shape: {current_dataframe.shape[0]} rows × {current_dataframe.shape[1]} columns
📝 Columns: {', '.join(current_dataframe.columns.tolist())}
            """
            
            return CallToolResult(
                content=[TextContent(type="text", text=info)]
            )
            
        except Exception as e:
            return CallToolResult(
                content=[TextContent(type="text", text=f"Error loading Excel file: {str(e)}")]
            )

    async def get_excel_info(self) -> CallToolResult:
        """Get information about the currently loaded Excel file."""
        global current_dataframe, current_excel_file
        
        if current_dataframe is None:
            return CallToolResult(
                content=[TextContent(type="text", text="No Excel file loaded. Please load a file first.")]
            )
        
        info = f"""
📊 **Excel File Information**
📁 File: {current_excel_file}
📈 Shape: {current_dataframe.shape[0]} rows × {current_dataframe.shape[1]} columns

📋 **Column Information:**
{current_dataframe.info()}

📊 **Data Types:**
{current_dataframe.dtypes.to_string()}

📈 **First 5 rows:**
{current_dataframe.head().to_string()}

📊 **Basic Statistics:**
{current_dataframe.describe().to_string()}
        """
        
        return CallToolResult(
            content=[TextContent(type="text", text=info)]
        )

    async def analyze_data(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Perform data analysis using local DeepSeek LLM."""
        global current_dataframe
        
        if current_dataframe is None:
            return CallToolResult(
                content=[TextContent(type="text", text="No Excel file loaded. Please load a file first.")]
            )
        
        analysis_type = arguments.get("analysis_type", "comprehensive")
        columns = arguments.get("columns", None)
        
        # Prepare data for analysis
        df_analysis = current_dataframe.copy()
        if columns:
            df_analysis = df_analysis[columns]
        
        # Generate analysis using local DeepSeek LLM
        analysis_result = await self._run_llm_analysis(df_analysis, analysis_type)
        
        return CallToolResult(
            content=[TextContent(type="text", text=analysis_result)]
        )

    async def create_visualization(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Create data visualizations."""
        global current_dataframe
        
        if current_dataframe is None:
            return CallToolResult(
                content=[TextContent(type="text", text="No Excel file loaded. Please load a file first.")]
            )
        
        chart_type = arguments["chart_type"]
        x_column = arguments.get("x_column")
        y_column = arguments.get("y_column")
        title = arguments.get("title", f"{chart_type.title()} Chart")
        
        try:
            # Create visualization
            fig = await self._create_chart(current_dataframe, chart_type, x_column, y_column, title)
            
            # Convert to base64 for embedding
            img_bytes = fig.to_image(format="png")
            img_base64 = base64.b64encode(img_bytes).decode()
            
            return CallToolResult(
                content=[
                    TextContent(type="text", text=f"✅ {title} created successfully!"),
                    ImageContent(
                        type="image",
                        data=EmbeddedResource(
                            uri=f"data:image/png;base64,{img_base64}",
                            mimeType="image/png"
                        )
                    )
                ]
            )
            
        except Exception as e:
            return CallToolResult(
                content=[TextContent(type="text", text=f"Error creating visualization: {str(e)}")]
            )

    async def generate_insights(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Generate business insights using DeepSeek LLM."""
        global current_dataframe
        
        if current_dataframe is None:
            return CallToolResult(
                content=[TextContent(type="text", text="No Excel file loaded. Please load a file first.")]
            )
        
        focus_area = arguments.get("focus_area", "general")
        business_context = arguments.get("business_context", "")
        
        # Generate insights using local DeepSeek LLM
        insights = await self._generate_llm_insights(current_dataframe, focus_area, business_context)
        
        return CallToolResult(
            content=[TextContent(type="text", text=insights)]
        )

    async def export_analysis(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Export analysis results."""
        global current_dataframe
        
        if current_dataframe is None:
            return CallToolResult(
                content=[TextContent(type="text", text="No Excel file loaded. Please load a file first.")]
            )
        
        export_format = arguments["format"]
        output_path = arguments.get("output_path", f"analysis_export.{export_format}")
        
        try:
            if export_format == "excel":
                current_dataframe.to_excel(output_path, index=False)
            elif export_format == "csv":
                current_dataframe.to_csv(output_path, index=False)
            elif export_format == "json":
                current_dataframe.to_json(output_path, orient="records", indent=2)
            elif export_format == "html":
                current_dataframe.to_html(output_path, index=False)
            
            return CallToolResult(
                content=[TextContent(type="text", text=f"✅ Analysis exported to {output_path}")]
            )
            
        except Exception as e:
            return CallToolResult(
                content=[TextContent(type="text", text=f"Error exporting analysis: {str(e)}")]
            )

    async def _run_llm_analysis(self, df: pd.DataFrame, analysis_type: str) -> str:
        """Run analysis using local DeepSeek LLM."""
        # This will be implemented in the LLM integration module
        from .llm_integration import analyze_with_deepseek
        
        return await analyze_with_deepseek(df, analysis_type)

    async def _create_chart(self, df: pd.DataFrame, chart_type: str, x_col: str, y_col: str, title: str):
        """Create various types of charts."""
        if chart_type == "histogram":
            fig = px.histogram(df, x=x_col, title=title)
        elif chart_type == "scatter":
            fig = px.scatter(df, x=x_col, y=y_col, title=title)
        elif chart_type == "line":
            fig = px.line(df, x=x_col, y=y_col, title=title)
        elif chart_type == "bar":
            fig = px.bar(df, x=x_col, y=y_col, title=title)
        elif chart_type == "correlation":
            corr_matrix = df.corr()
            fig = px.imshow(corr_matrix, title=title)
        elif chart_type == "boxplot":
            fig = px.box(df, x=x_col, y=y_col, title=title)
        else:
            raise ValueError(f"Unsupported chart type: {chart_type}")
        
        return fig

    async def _generate_llm_insights(self, df: pd.DataFrame, focus_area: str, business_context: str) -> str:
        """Generate insights using local DeepSeek LLM."""
        from .llm_integration import generate_insights_with_deepseek
        
        return await generate_insights_with_deepseek(df, focus_area, business_context)

async def main():
    """Main entry point."""
    server = ExcelAnalyzerMCPServer()
    
    # Run the server
    async with stdio_server() as (read_stream, write_stream):
        await server.server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="excel-analyzer",
                server_version="1.0.0",
                capabilities=server.server.get_capabilities(
                    notification_options=None,
                    experimental_capabilities=None,
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main()) 