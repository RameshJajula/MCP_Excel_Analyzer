# mcp_server.py — MCP Excel Analyzer

This file implements the **Model Context Protocol (MCP) server** for Excel data analysis using a local DeepSeek LLM. It is the heart of the system, orchestrating all tool registration, Excel file handling, LLM-powered analysis, visualization, and export.

---

## 📄 **File Overview**
- **Purpose:** Expose a set of tools (load, analyze, visualize, export, etc.) via MCP for Excel data analysis.
- **Main Class:** `ExcelAnalyzerMCPServer`
- **Interacts With:**
  - `llm_integration.py` for LLM analysis/insights
  - `pandas` for Excel file handling
  - `plotly` for visualization

---

## 🧩 **Key Components & Functions**

### `ExcelAnalyzerMCPServer`
The main class that sets up the MCP server and registers all tools.

#### `__init__(self)`
- Initializes the MCP server and registers tools.
- Calls `self.setup_tools()`.

#### `setup_tools(self)`
- Registers all available tools with the MCP server.
- Each tool is defined with a name, description, and input schema.
- Tools include:
  - `load_excel_file`: Load an Excel file for analysis.
  - `get_excel_info`: Get info about the loaded file.
  - `analyze_data`: Run LLM-powered analysis.
  - `create_visualization`: Generate charts.
  - `generate_insights`: Business insights via LLM.
  - `export_analysis`: Export results.

#### `handle_list_tools()`
- Returns a list of all registered tools and their schemas.
- Used by MCP clients to discover available actions.

#### `handle_call_tool(name, arguments)`
- Main dispatcher for tool calls.
- Based on the tool name, calls the appropriate method (e.g., `load_excel_file`, `analyze_data`, etc.).
- Handles errors and returns results in MCP format.

#### `load_excel_file(self, arguments)`
- Loads an Excel file from the given path.
- Reads the first sheet into a pandas DataFrame.
- Stores the DataFrame and file path globally.
- Returns info about the file (shape, columns, etc.).

#### `get_excel_info(self)`
- Returns detailed info about the currently loaded DataFrame (shape, columns, dtypes, head, stats).

#### `analyze_data(self, arguments)`
- Prepares the DataFrame (optionally filters columns).
- Calls the LLM integration (`llm_integration.analyze_with_deepseek`) to generate analysis.
- Returns the LLM's analysis as text.

#### `create_visualization(self, arguments)`
- Generates a chart (histogram, scatter, line, bar, correlation, boxplot) using Plotly.
- Returns the chart as a base64-encoded image.

#### `generate_insights(self, arguments)`
- Calls the LLM integration to generate business insights based on the DataFrame and user context.

#### `export_analysis(self, arguments)`
- Exports the current DataFrame to Excel, CSV, JSON, or HTML.
- Saves the file to the specified path.

#### `_run_llm_analysis(self, df, analysis_type)`
- Helper to call the LLM for data analysis.

#### `_create_chart(self, df, chart_type, x_col, y_col, title)`
- Helper to generate Plotly charts.

#### `_generate_llm_insights(self, df, focus_area, business_context)`
- Helper to call the LLM for business insights.

#### `main()`
- Entry point to start the MCP server using stdio.

---

## 🔄 **Code Flow Walkthrough**

1. **Server Initialization:**
   - `ExcelAnalyzerMCPServer` is instantiated.
   - Tools are registered via `setup_tools()`.
2. **Tool Discovery:**
   - MCP clients call `list_tools` to see available actions.
3. **Tool Invocation:**
   - MCP clients call `call_tool` with a tool name and arguments.
   - The server dispatches to the correct method.
4. **Excel File Handling:**
   - `load_excel_file` loads the file and stores it as a DataFrame.
   - `get_excel_info` provides metadata and stats.
5. **Analysis/Insights:**
   - `analyze_data` and `generate_insights` call the LLM integration for advanced analysis.
6. **Visualization:**
   - `create_visualization` generates and returns charts.
7. **Export:**
   - `export_analysis` saves the DataFrame in the requested format.

---

## 🤝 **Inter-module Interactions**
- **LLM Integration:** Calls `llm_integration.py` for all LLM-powered features.
- **Web Interface:** Can be used as a backend for the web UI or other MCP clients.

---

## 📝 **Tips for Reading the Code**
- Start with `ExcelAnalyzerMCPServer` and its methods.
- Follow the tool registration and dispatch logic.
- See how each tool method prepares data and calls helpers or LLM integration.
- Use this file as a reference for adding new tools or extending functionality. 