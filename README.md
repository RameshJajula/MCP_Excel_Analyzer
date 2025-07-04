# MCP Excel Analyzer

A powerful, extensible platform for analyzing local Excel files using a local DeepSeek LLM (Large Language Model), orchestrated via the Model Context Protocol (MCP). This project provides both a programmatic MCP server and a beautiful web interface for business and data analysis workflows.

---

## 🚀 **Project Architecture Overview**

```
MCP Excel Analyzer
│
├── mcp_server.py         # MCP server: protocol, tool registry, Excel/LLM orchestration
├── llm_integration.py    # DeepSeek LLM integration: prompt engineering, model inference
├── web_interface.py      # FastAPI web UI: upload, analyze, visualize, export
├── __init__.py           # Package init, exports
├── requirements.txt      # All dependencies
└── README.md             # This documentation
```

---

## 🏗️ **Architecture Diagram**

```mermaid
flowchart TD
    A[User (Web/CLI)] -->|Upload Excel, Request Analysis| B(Web Interface)
    B -->|API Calls| C(MCP Server)
    C -->|DataFrame, Prompts| D(DeepSeek LLM)
    D -->|Insights, Analysis| C
    C -->|Results, Charts| B
    B -->|Download, View| A
    C -->|Reads/Writes| E[Excel/CSV Files]
```

---

## 📂 **File-by-File Logic**

### 1. `mcp_server.py`  
**Role:** The heart of the system. Implements the MCP server, registers all tools, and orchestrates Excel file loading, analysis, visualization, and export.

- **Tool Registration:** Defines tools (load, info, analyze, visualize, export, insights) and their schemas.
- **Excel Handling:** Loads Excel files, parses sheets, manages current DataFrame.
- **LLM Orchestration:** Calls LLM integration for analysis/insights.
- **Visualization:** Uses Plotly for chart generation, returns images as base64.
- **Export:** Supports Excel, CSV, JSON, HTML exports.
- **Protocol:** Implements MCP server methods for tool listing and invocation.

### 2. `llm_integration.py`  
**Role:** Handles all interaction with the local DeepSeek LLM.

- **Model Loading:** Loads DeepSeek model from local path or HuggingFace.
- **Prompt Engineering:** Summarizes DataFrame, crafts prompts for different analysis types.
- **LLM Inference:** Runs model to generate analysis/insights.
- **Fallback:** Provides a simple analyzer if LLM is unavailable.
- **Async:** All model calls are async for performance.

### 3. `web_interface.py`  
**Role:** Provides a modern, user-friendly web interface using FastAPI.

- **Upload:** Drag-and-drop or browse to upload Excel/CSV files.
- **Analysis:** Select analysis type, run LLM-powered analysis.
- **Visualization:** Choose chart type, render interactive Plotly charts.
- **Insights:** Enter business context, get actionable LLM insights.
- **Export:** Download results in multiple formats.
- **Frontend:** Beautiful HTML/CSS/JS for business users.

### 4. `__init__.py`  
**Role:** Package initialization and exports.

- Exports main classes and functions for programmatic use.

### 5. `requirements.txt`  
**Role:** Lists all dependencies for the project.

- pandas, openpyxl, plotly, transformers, torch, fastapi, uvicorn, etc.

---

## 🛠️ **How to Use**

### 1. **Install Dependencies**

```bash
pip install -r requirements.txt
```

### 2. **Run the Web Interface**

```bash
python web_interface.py
```
- Open your browser at [http://localhost:8000](http://localhost:8000)

### 3. **Or Run the MCP Server Directly**

```bash
python mcp_server.py
```
- Integrate with other MCP clients or tools.

---

## 🧠 **How It Works**

1. **Upload Excel File:** User uploads a file via web or API.
2. **File Loaded:** Server parses file, stores as pandas DataFrame.
3. **Analysis/Visualization:** User selects analysis or chart type.
4. **LLM Prompt:** DataFrame is summarized and sent as a prompt to DeepSeek LLM.
5. **LLM Response:** Model returns insights, which are displayed or exported.
6. **Export:** User can download results in various formats.

---

## ⚡ **Quickstart Example**

### Web Interface Walkthrough

1. **Start the Web Server:**
   ```bash
   python web_interface.py
   ```
2. **Open your browser:** Go to [http://localhost:8000](http://localhost:8000)
3. **Upload an Excel/CSV file:** Drag and drop or click to select your file.
4. **Explore:**
   - **Data Analysis:** Choose analysis type (basic, statistical, comprehensive, etc.) and click "Analyze Data".
   - **Visualizations:** Select chart type and generate interactive charts.
   - **Business Insights:** Enter a focus area and context, then generate LLM-powered insights.
   - **Export:** Download results in Excel, CSV, JSON, or HTML.

**Example Screenshot:**

> ![Web UI Example](https://user-images.githubusercontent.com/placeholder/web-ui-example.png)

### MCP Server (Programmatic) Example

You can use the MCP server in your own Python code or via CLI tools that support MCP.

```python
from mcp_excel_analyzer.mcp_server import ExcelAnalyzerMCPServer
import asyncio

async def main():
    server = ExcelAnalyzerMCPServer()
    # Use server.server.call_tool(...) to invoke tools programmatically
    # Example: load an Excel file
    result = await server.load_excel_file({"file_path": "./your_data.xlsx"})
    print(result.content[0].text)

asyncio.run(main())
```

---

## 🔌 **Customization & Extension**

- **Add More Tools:** Extend `mcp_server.py` with new analysis or business logic.
- **Change Model:** Edit `llm_integration.py` to use a different LLM or prompt style.
- **UI Tweaks:** Modify `web_interface.py` for branding or new features.

---

## 🛡️ **Security & Performance**

- **Local Only:** By default, runs on localhost. Do not expose to the public internet without authentication.
- **Large Files:** For very large Excel files, consider chunked reading or server-side pagination.
- **GPU Support:** DeepSeek LLM will use GPU if available for faster inference.

---

## 🤝 **Contributing**

Pull requests and issues are welcome! Please open an issue for feature requests or bug reports.

---

## 📞 **Support**

For help, open an issue or contact the maintainers. 