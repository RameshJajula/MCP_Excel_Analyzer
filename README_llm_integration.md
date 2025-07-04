# llm_integration.py — MCP Excel Analyzer

This file handles all integration with the local DeepSeek LLM (Large Language Model) for data analysis and business insights. It provides prompt engineering, model loading, inference, and fallback logic.

---

## 📄 **File Overview**
- **Purpose:** Interface with DeepSeek LLM for advanced data analysis and insight generation.
- **Main Class:** `DeepSeekAnalyzer`
- **Interacts With:**
  - `mcp_server.py` (calls analysis/insight functions)
  - `transformers`, `torch` (for LLM inference)
  - `pandas` (for DataFrame summaries)

---

## 🧩 **Key Components & Functions**

### `DeepSeekAnalyzer`
Main class for managing the DeepSeek LLM.

#### `__init__(self, model_path)`
- Sets up model path, device, and state.

#### `load_model(self)`
- Loads the DeepSeek model and tokenizer from local path or HuggingFace.
- Uses GPU if available.

#### `_prepare_data_summary(self, df)`
- Summarizes a pandas DataFrame (shape, columns, dtypes, missing values, stats, correlations) for use in LLM prompts.

#### `_create_analysis_prompt(self, df, analysis_type)`
- Crafts a prompt for the LLM based on the analysis type (basic, statistical, correlation, trends, outliers, comprehensive).

#### `_create_insights_prompt(self, df, focus_area, business_context)`
- Crafts a prompt for the LLM to generate business insights.

#### `_generate_response(self, prompt, max_length)`
- Runs the LLM to generate a response to the prompt.
- Handles tokenization, inference, and decoding.

#### `analyze_data(self, df, analysis_type)`
- High-level method to analyze data using the LLM.
- Calls prompt creation and response generation.

#### `generate_insights(self, df, focus_area, business_context)`
- High-level method to generate business insights using the LLM.

### Global Functions

#### `get_analyzer()`
- Returns a singleton instance of `DeepSeekAnalyzer`.

#### `analyze_with_deepseek(df, analysis_type)`
- Async function to analyze data using the LLM (with fallback to simple analyzer).

#### `generate_insights_with_deepseek(df, focus_area, business_context)`
- Async function to generate insights using the LLM (with fallback).

### `SimpleAnalyzer`
- Provides basic, non-LLM analysis and insights for fallback/testing.

---

## 🔄 **Code Flow Walkthrough**

1. **Initialization:**
   - `DeepSeekAnalyzer` is created (singleton).
   - Model is loaded on first use.
2. **Prompt Engineering:**
   - DataFrame is summarized and prompt is crafted for the requested analysis/insight.
3. **LLM Inference:**
   - Prompt is tokenized and sent to the model.
   - Model generates a response, which is decoded and returned.
4. **Fallback:**
   - If LLM is unavailable, `SimpleAnalyzer` provides basic analysis/insights.
5. **Integration:**
   - `mcp_server.py` calls `analyze_with_deepseek` and `generate_insights_with_deepseek` for all LLM-powered features.

---

## 🤝 **Inter-module Interactions**
- **MCP Server:** All LLM-powered features in `mcp_server.py` use this module.
- **pandas DataFrames:** All analysis is based on DataFrame summaries.

---

## 📝 **Tips for Reading the Code**
- Start with `DeepSeekAnalyzer` and its methods.
- See how prompts are constructed and how the model is called.
- Review the fallback logic for non-LLM analysis.
- Use this file as a reference for customizing LLM prompts or swapping models. 