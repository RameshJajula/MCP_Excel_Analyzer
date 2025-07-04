#!/usr/bin/env python3
"""
Web Interface for MCP Excel Analyzer
Provides a user-friendly web interface for Excel data analysis
"""

import os
import asyncio
import logging
from typing import Optional, List, Dict, Any
from pathlib import Path
import tempfile
import shutil

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

from .llm_integration import analyze_with_deepseek, generate_insights_with_deepseek

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
current_dataframe: Optional[pd.DataFrame] = None
current_file_path: Optional[str] = None

app = FastAPI(
    title="MCP Excel Analyzer",
    description="Web interface for Excel data analysis with local DeepSeek LLM",
    version="1.0.0"
)

# Create static directory for temporary files
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def home():
    """Home page with file upload and analysis interface."""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MCP Excel Analyzer</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .header h1 {
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }
        .header p {
            margin: 10px 0 0 0;
            opacity: 0.9;
        }
        .content {
            padding: 30px;
        }
        .upload-section {
            border: 2px dashed #ddd;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            margin-bottom: 30px;
            transition: all 0.3s ease;
        }
        .upload-section:hover {
            border-color: #667eea;
            background: #f8f9ff;
        }
        .upload-section.dragover {
            border-color: #667eea;
            background: #f0f4ff;
        }
        .file-input {
            display: none;
        }
        .upload-btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 16px;
            transition: all 0.3s ease;
        }
        .upload-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
        }
        .analysis-section {
            display: none;
            margin-top: 30px;
        }
        .analysis-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .analysis-card {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            border-left: 4px solid #667eea;
        }
        .analysis-card h3 {
            margin: 0 0 15px 0;
            color: #333;
        }
        .btn {
            background: #667eea;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            margin: 5px;
            transition: all 0.3s ease;
        }
        .btn:hover {
            background: #5a6fd8;
            transform: translateY(-1px);
        }
        .btn-secondary {
            background: #6c757d;
        }
        .btn-secondary:hover {
            background: #5a6268;
        }
        .results-section {
            margin-top: 30px;
        }
        .result-card {
            background: white;
            border: 1px solid #ddd;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .loading {
            text-align: center;
            padding: 40px;
            color: #666;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .file-info {
            background: #e8f4fd;
            border: 1px solid #bee5eb;
            border-radius: 5px;
            padding: 15px;
            margin: 20px 0;
        }
        .chart-container {
            margin: 20px 0;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 10px;
        }
        .insights-form {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
        }
        .form-group {
            margin-bottom: 15px;
        }
        .form-group label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
            color: #333;
        }
        .form-group input, .form-group textarea, .form-group select {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 14px;
        }
        .form-group textarea {
            height: 100px;
            resize: vertical;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 MCP Excel Analyzer</h1>
            <p>Analyze Excel data with local DeepSeek LLM using Model Context Protocol</p>
        </div>
        
        <div class="content">
            <div class="upload-section" id="uploadSection">
                <h2>📁 Upload Excel File</h2>
                <p>Drag and drop your Excel file here or click to browse</p>
                <input type="file" id="fileInput" class="file-input" accept=".xlsx,.xls,.csv">
                <button class="upload-btn" onclick="document.getElementById('fileInput').click()">
                    Choose File
                </button>
            </div>
            
            <div class="analysis-section" id="analysisSection">
                <div class="file-info" id="fileInfo"></div>
                
                <div class="analysis-grid">
                    <div class="analysis-card">
                        <h3>📈 Data Analysis</h3>
                        <p>Perform comprehensive data analysis using DeepSeek LLM</p>
                        <select id="analysisType" class="form-control">
                            <option value="basic">Basic Analysis</option>
                            <option value="statistical">Statistical Analysis</option>
                            <option value="correlation">Correlation Analysis</option>
                            <option value="trends">Trend Analysis</option>
                            <option value="outliers">Outlier Detection</option>
                            <option value="comprehensive" selected>Comprehensive Analysis</option>
                        </select>
                        <button class="btn" onclick="analyzeData()">Analyze Data</button>
                    </div>
                    
                    <div class="analysis-card">
                        <h3>📊 Visualizations</h3>
                        <p>Create charts and graphs from your data</p>
                        <select id="chartType" class="form-control">
                            <option value="histogram">Histogram</option>
                            <option value="scatter">Scatter Plot</option>
                            <option value="line">Line Chart</option>
                            <option value="bar">Bar Chart</option>
                            <option value="correlation">Correlation Matrix</option>
                            <option value="boxplot">Box Plot</option>
                        </select>
                        <button class="btn" onclick="createVisualization()">Create Chart</button>
                    </div>
                    
                    <div class="analysis-card">
                        <h3>💡 Business Insights</h3>
                        <p>Generate actionable business insights</p>
                        <button class="btn" onclick="showInsightsForm()">Generate Insights</button>
                    </div>
                    
                    <div class="analysis-card">
                        <h3>💾 Export Results</h3>
                        <p>Export analysis results in various formats</p>
                        <select id="exportFormat" class="form-control">
                            <option value="excel">Excel</option>
                            <option value="csv">CSV</option>
                            <option value="json">JSON</option>
                            <option value="html">HTML</option>
                        </select>
                        <button class="btn" onclick="exportAnalysis()">Export</button>
                    </div>
                </div>
                
                <div class="insights-form" id="insightsForm" style="display: none;">
                    <h3>💡 Generate Business Insights</h3>
                    <div class="form-group">
                        <label for="focusArea">Focus Area:</label>
                        <input type="text" id="focusArea" placeholder="e.g., sales, performance, trends">
                    </div>
                    <div class="form-group">
                        <label for="businessContext">Business Context:</label>
                        <textarea id="businessContext" placeholder="Provide additional business context for better insights..."></textarea>
                    </div>
                    <button class="btn" onclick="generateInsights()">Generate Insights</button>
                    <button class="btn btn-secondary" onclick="hideInsightsForm()">Cancel</button>
                </div>
                
                <div class="results-section" id="resultsSection"></div>
            </div>
        </div>
    </div>

    <script>
        let currentData = null;
        let currentColumns = [];
        
        // File upload handling
        document.getElementById('fileInput').addEventListener('change', handleFileSelect);
        
        const uploadSection = document.getElementById('uploadSection');
        uploadSection.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadSection.classList.add('dragover');
        });
        
        uploadSection.addEventListener('dragleave', () => {
            uploadSection.classList.remove('dragover');
        });
        
        uploadSection.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadSection.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFile(files[0]);
            }
        });
        
        function handleFileSelect(event) {
            const file = event.target.files[0];
            if (file) {
                handleFile(file);
            }
        }
        
        async function handleFile(file) {
            const formData = new FormData();
            formData.append('file', file);
            
            try {
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (response.ok) {
                    currentData = result.data;
                    currentColumns = result.columns;
                    showAnalysisSection(result.file_info);
                } else {
                    alert('Error: ' + result.detail);
                }
            } catch (error) {
                alert('Error uploading file: ' + error.message);
            }
        }
        
        function showAnalysisSection(fileInfo) {
            document.getElementById('fileInfo').innerHTML = fileInfo;
            document.getElementById('analysisSection').style.display = 'block';
        }
        
        async function analyzeData() {
            const analysisType = document.getElementById('analysisType').value;
            showLoading('Analyzing data with DeepSeek LLM...');
            
            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        analysis_type: analysisType
                    })
                });
                
                const result = await response.json();
                
                if (response.ok) {
                    showResult('Data Analysis Results', result.analysis, 'analysis');
                } else {
                    showError('Analysis failed: ' + result.detail);
                }
            } catch (error) {
                showError('Error during analysis: ' + error.message);
            }
        }
        
        async function createVisualization() {
            const chartType = document.getElementById('chartType').value;
            showLoading('Creating visualization...');
            
            try {
                const response = await fetch('/visualize', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        chart_type: chartType
                    })
                });
                
                const result = await response.json();
                
                if (response.ok) {
                    showChart(result.chart_data, result.title);
                } else {
                    showError('Visualization failed: ' + result.detail);
                }
            } catch (error) {
                showError('Error creating visualization: ' + error.message);
            }
        }
        
        function showInsightsForm() {
            document.getElementById('insightsForm').style.display = 'block';
        }
        
        function hideInsightsForm() {
            document.getElementById('insightsForm').style.display = 'none';
        }
        
        async function generateInsights() {
            const focusArea = document.getElementById('focusArea').value;
            const businessContext = document.getElementById('businessContext').value;
            
            if (!focusArea.trim()) {
                alert('Please enter a focus area');
                return;
            }
            
            showLoading('Generating business insights with DeepSeek LLM...');
            
            try {
                const response = await fetch('/insights', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        focus_area: focusArea,
                        business_context: businessContext
                    })
                });
                
                const result = await response.json();
                
                if (response.ok) {
                    showResult('Business Insights', result.insights, 'insights');
                    hideInsightsForm();
                } else {
                    showError('Insights generation failed: ' + result.detail);
                }
            } catch (error) {
                showError('Error generating insights: ' + error.message);
            }
        }
        
        async function exportAnalysis() {
            const format = document.getElementById('exportFormat').value;
            
            try {
                const response = await fetch('/export', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        format: format
                    })
                });
                
                if (response.ok) {
                    const blob = await response.blob();
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `analysis_export.${format}`;
                    a.click();
                    window.URL.revokeObjectURL(url);
                } else {
                    const result = await response.json();
                    alert('Export failed: ' + result.detail);
                }
            } catch (error) {
                alert('Error exporting: ' + error.message);
            }
        }
        
        function showLoading(message) {
            const resultsSection = document.getElementById('resultsSection');
            resultsSection.innerHTML = `
                <div class="loading">
                    <div class="spinner"></div>
                    <p>${message}</p>
                </div>
            `;
        }
        
        function showResult(title, content, type) {
            const resultsSection = document.getElementById('resultsSection');
            const resultCard = document.createElement('div');
            resultCard.className = 'result-card';
            
            if (type === 'analysis' || type === 'insights') {
                resultCard.innerHTML = `
                    <h3>${title}</h3>
                    <div style="white-space: pre-wrap; font-family: monospace; background: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto;">
                        ${content}
                    </div>
                `;
            } else {
                resultCard.innerHTML = `
                    <h3>${title}</h3>
                    <div>${content}</div>
                `;
            }
            
            resultsSection.appendChild(resultCard);
        }
        
        function showChart(chartData, title) {
            const resultsSection = document.getElementById('resultsSection');
            const resultCard = document.createElement('div');
            resultCard.className = 'result-card';
            
            resultCard.innerHTML = `
                <h3>${title}</h3>
                <div id="chart-${Date.now()}" class="chart-container"></div>
            `;
            
            resultsSection.appendChild(resultCard);
            
            // Render the chart
            const chartDiv = resultCard.querySelector(`#chart-${Date.now()}`);
            Plotly.newPlot(chartDiv, chartData.data, chartData.layout);
        }
        
        function showError(message) {
            const resultsSection = document.getElementById('resultsSection');
            const resultCard = document.createElement('div');
            resultCard.className = 'result-card';
            resultCard.style.borderLeftColor = '#dc3545';
            resultCard.innerHTML = `
                <h3 style="color: #dc3545;">Error</h3>
                <p>${message}</p>
            `;
            resultsSection.appendChild(resultCard);
        }
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Handle file upload."""
    global current_dataframe, current_file_path
    
    try:
        # Save uploaded file temporarily
        temp_dir = Path("static/temp")
        temp_dir.mkdir(exist_ok=True)
        
        file_path = temp_dir / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Read Excel file
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        
        current_dataframe = df
        current_file_path = str(file_path)
        
        file_info = f"""
        <strong>File loaded successfully!</strong><br>
        <strong>Filename:</strong> {file.filename}<br>
        <strong>Shape:</strong> {df.shape[0]} rows × {df.shape[1]} columns<br>
        <strong>Columns:</strong> {', '.join(df.columns.tolist())}<br>
        <strong>Memory Usage:</strong> {df.memory_usage(deep=True).sum() / 1024:.2f} KB
        """
        
        return JSONResponse(content={
            "message": "File uploaded successfully",
            "file_info": file_info,
            "data": df.head(10).to_dict('records'),
            "columns": df.columns.tolist()
        })
        
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/analyze")
async def analyze_data(request: Dict[str, Any]):
    """Analyze data using DeepSeek LLM."""
    global current_dataframe
    
    if current_dataframe is None:
        raise HTTPException(status_code=400, detail="No file loaded")
    
    try:
        analysis_type = request.get("analysis_type", "comprehensive")
        analysis_result = await analyze_with_deepseek(current_dataframe, analysis_type)
        
        return JSONResponse(content={
            "analysis": analysis_result
        })
        
    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/visualize")
async def create_visualization(request: Dict[str, Any]):
    """Create data visualization."""
    global current_dataframe
    
    if current_dataframe is None:
        raise HTTPException(status_code=400, detail="No file loaded")
    
    try:
        chart_type = request["chart_type"]
        x_column = request.get("x_column")
        y_column = request.get("y_column")
        title = request.get("title", f"{chart_type.title()} Chart")
        
        # Create chart based on type
        if chart_type == "histogram":
            if not x_column:
                x_column = current_dataframe.select_dtypes(include=['number']).columns[0]
            fig = px.histogram(current_dataframe, x=x_column, title=title)
        elif chart_type == "scatter":
            if not x_column or not y_column:
                numeric_cols = current_dataframe.select_dtypes(include=['number']).columns
                if len(numeric_cols) >= 2:
                    x_column = numeric_cols[0]
                    y_column = numeric_cols[1]
                else:
                    raise HTTPException(status_code=400, detail="Need at least 2 numeric columns for scatter plot")
            fig = px.scatter(current_dataframe, x=x_column, y=y_column, title=title)
        elif chart_type == "line":
            if not x_column or not y_column:
                numeric_cols = current_dataframe.select_dtypes(include=['number']).columns
                if len(numeric_cols) >= 2:
                    x_column = numeric_cols[0]
                    y_column = numeric_cols[1]
                else:
                    raise HTTPException(status_code=400, detail="Need at least 2 numeric columns for line chart")
            fig = px.line(current_dataframe, x=x_column, y=y_column, title=title)
        elif chart_type == "bar":
            if not x_column or not y_column:
                numeric_cols = current_dataframe.select_dtypes(include=['number']).columns
                if len(numeric_cols) >= 2:
                    x_column = numeric_cols[0]
                    y_column = numeric_cols[1]
                else:
                    raise HTTPException(status_code=400, detail="Need at least 2 numeric columns for bar chart")
            fig = px.bar(current_dataframe, x=x_column, y=y_column, title=title)
        elif chart_type == "correlation":
            corr_matrix = current_dataframe.select_dtypes(include=['number']).corr()
            fig = px.imshow(corr_matrix, title=title)
        elif chart_type == "boxplot":
            if not x_column or not y_column:
                numeric_cols = current_dataframe.select_dtypes(include=['number']).columns
                if len(numeric_cols) >= 2:
                    x_column = numeric_cols[0]
                    y_column = numeric_cols[1]
                else:
                    raise HTTPException(status_code=400, detail="Need at least 2 numeric columns for box plot")
            fig = px.box(current_dataframe, x=x_column, y=y_column, title=title)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported chart type: {chart_type}")
        
        return JSONResponse(content={
            "chart_data": {
                "data": fig.data,
                "layout": fig.layout
            },
            "title": title
        })
        
    except Exception as e:
        logger.error(f"Error creating visualization: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/insights")
async def generate_insights(request: Dict[str, Any]):
    """Generate business insights."""
    global current_dataframe
    
    if current_dataframe is None:
        raise HTTPException(status_code=400, detail="No file loaded")
    
    try:
        focus_area = request.get("focus_area", "general")
        business_context = request.get("business_context", "")
        
        insights = await generate_insights_with_deepseek(
            current_dataframe, focus_area, business_context
        )
        
        return JSONResponse(content={
            "insights": insights
        })
        
    except Exception as e:
        logger.error(f"Error generating insights: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/export")
async def export_analysis(request: Dict[str, Any]):
    """Export analysis results."""
    global current_dataframe
    
    if current_dataframe is None:
        raise HTTPException(status_code=400, detail="No file loaded")
    
    try:
        export_format = request["format"]
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{export_format}") as tmp_file:
            if export_format == "excel":
                current_dataframe.to_excel(tmp_file.name, index=False)
            elif export_format == "csv":
                current_dataframe.to_csv(tmp_file.name, index=False)
            elif export_format == "json":
                current_dataframe.to_json(tmp_file.name, orient="records", indent=2)
            elif export_format == "html":
                current_dataframe.to_html(tmp_file.name, index=False)
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported format: {export_format}")
            
            # Read file and return as response
            with open(tmp_file.name, 'rb') as f:
                content = f.read()
            
            # Clean up
            os.unlink(tmp_file.name)
            
            return JSONResponse(
                content=content,
                media_type=f"application/{export_format}",
                headers={"Content-Disposition": f"attachment; filename=analysis_export.{export_format}"}
            )
        
    except Exception as e:
        logger.error(f"Error exporting analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def run_web_interface(host: str = "127.0.0.1", port: int = 8000):
    """Run the web interface."""
    logger.info(f"Starting web interface at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    run_web_interface() 