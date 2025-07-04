# __init__.py — MCP Excel Analyzer

This file is the package initializer for the `mcp_excel_analyzer` module. It makes the main classes and functions importable from the package and defines package metadata.

---

## 📄 **File Overview**
- **Purpose:** Initialize the package, set metadata, and export key classes/functions.
- **Interacts With:**
  - `mcp_server.py` (main server class)
  - `llm_integration.py` (LLM integration functions)

---

## 🧩 **Key Components**

### Metadata
- `__version__`: Package version.
- `__author__`: Author or team name.

### Exports
- `ExcelAnalyzerMCPServer`: Main MCP server class.
- `DeepSeekAnalyzer`: LLM integration class.
- `analyze_with_deepseek`, `generate_insights_with_deepseek`: Async functions for LLM-powered analysis/insights.

### `__all__`
- Defines the public API of the package (what is importable with `from mcp_excel_analyzer import *`).

---

## 🔄 **Code Flow Walkthrough**
- When you import from `mcp_excel_analyzer`, this file ensures you get the main server and LLM integration tools.
- Keeps the package organized and easy to use in other projects.

---

## 📝 **Tips for Reading the Code**
- Use this file as a reference for what the package exposes.
- If you add new main features/classes, export them here for easy access. 