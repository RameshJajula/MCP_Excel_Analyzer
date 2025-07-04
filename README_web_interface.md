# web_interface.py — MCP Excel Analyzer

This file provides a modern, user-friendly web interface for the MCP Excel Analyzer using FastAPI. It allows users to upload Excel files, run LLM-powered analysis, create visualizations, generate business insights, and export results—all from their browser.

---

## 📄 **File Overview**
- **Purpose:** Expose all MCP Excel Analyzer features via a web UI.
- **Framework:** FastAPI (backend), HTML/CSS/JS (frontend)
- **Interacts With:**
  - `llm_integration.py` for LLM analysis/insights
  - `pandas` for DataFrame handling
  - `plotly` for charting

---

## 🧩 **Key Components & Functions**

### FastAPI App (`app`)
- Main FastAPI application instance.

### Routes

#### `GET /`
- Serves the main HTML page (file upload, analysis, visualization, export UI).

#### `POST /upload`
- Handles Excel/CSV file uploads.
- Saves file, loads into pandas DataFrame, returns file info and preview.

#### `POST /analyze`
- Runs LLM-powered data analysis on the uploaded DataFrame.
- Accepts analysis type (basic, statistical, etc.).
- Returns analysis text.

#### `POST /visualize`
- Generates a chart (histogram, scatter, line, bar, correlation, boxplot) using Plotly.
- Returns chart data for rendering in the browser.

#### `POST /insights`
- Runs LLM-powered business insights generation.
- Accepts focus area and business context.
- Returns insights text.

#### `POST /export`
- Exports the current DataFrame in Excel, CSV, JSON, or HTML format.
- Returns the file for download.

### `run_web_interface()`
- Starts the FastAPI server using Uvicorn.

---

## 🔄 **Code Flow Walkthrough**

1. **Startup:**
   - FastAPI app is created and static directory is set up.
2. **User Interaction:**
   - User opens the web page, uploads a file, and interacts with the UI.
3. **File Upload:**
   - `/upload` saves the file and loads it into a DataFrame.
4. **Analysis/Visualization:**
   - `/analyze` and `/visualize` call LLM and charting functions.
5. **Insights:**
   - `/insights` calls LLM for business recommendations.
6. **Export:**
   - `/export` lets the user download results in various formats.

---

## 🤝 **Inter-module Interactions**
- **LLM Integration:** Calls `llm_integration.py` for all LLM-powered features.
- **pandas DataFrames:** All analysis and visualization is based on the loaded DataFrame.

---

## 📝 **Tips for Reading the Code**
- Start with the route definitions to see the user flow.
- Review how each route interacts with the DataFrame and LLM integration.
- Use this file as a reference for customizing the web UI or adding new features. 