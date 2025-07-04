#!/usr/bin/env python3
"""
DeepSeek LLM Integration for Excel Data Analysis
This module handles the integration with local DeepSeek language models
for performing data analysis and generating insights.
"""

import asyncio
import json
import logging
import os
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.nn import functional as F

logger = logging.getLogger(__name__)

class DeepSeekAnalyzer:
    def __init__(self, model_path: str = "deepseek-ai/deepseek-coder-6.7b-instruct"):
        """
        Initialize the DeepSeek analyzer.
        
        Args:
            model_path: Path to the local DeepSeek model or HuggingFace model name
        """
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.is_loaded = False
        
    async def load_model(self):
        """Load the DeepSeek model asynchronously."""
        if self.is_loaded:
            return
            
        try:
            logger.info(f"Loading DeepSeek model from {self.model_path}...")
            logger.info(f"Using device: {self.device}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            self.is_loaded = True
            logger.info("DeepSeek model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading DeepSeek model: {str(e)}")
            raise
    
    def _prepare_data_summary(self, df: pd.DataFrame) -> str:
        """Prepare a comprehensive data summary for the LLM."""
        summary = f"""
Dataset Summary:
- Shape: {df.shape[0]} rows × {df.shape[1]} columns
- Columns: {', '.join(df.columns.tolist())}
- Data types: {dict(df.dtypes)}
- Missing values: {df.isnull().sum().to_dict()}
- Memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB

First 10 rows:
{df.head(10).to_string()}

Basic statistics:
{df.describe().to_string()}

Correlation matrix (numeric columns only):
{df.select_dtypes(include=[np.number]).corr().to_string()}
        """
        return summary
    
    def _create_analysis_prompt(self, df: pd.DataFrame, analysis_type: str) -> str:
        """Create a prompt for data analysis."""
        data_summary = self._prepare_data_summary(df)
        
        prompts = {
            "basic": f"""
You are a data analyst. Analyze the following dataset and provide basic insights:

{data_summary}

Please provide:
1. Key observations about the data
2. Data quality assessment
3. Potential issues or anomalies
4. Recommendations for further analysis

Respond in a clear, structured format.
            """,
            
            "statistical": f"""
You are a statistical analyst. Perform a comprehensive statistical analysis of this dataset:

{data_summary}

Please provide:
1. Descriptive statistics interpretation
2. Distribution analysis for key variables
3. Statistical significance of patterns
4. Confidence intervals where applicable
5. Statistical insights and implications

Use statistical terminology and provide detailed analysis.
            """,
            
            "correlation": f"""
You are a correlation analyst. Analyze the relationships between variables in this dataset:

{data_summary}

Please provide:
1. Correlation analysis between numeric variables
2. Strength and direction of relationships
3. Potential causal relationships
4. Multicollinearity assessment
5. Recommendations based on correlations

Focus on identifying meaningful relationships in the data.
            """,
            
            "trends": f"""
You are a trend analyst. Identify and analyze trends in this dataset:

{data_summary}

Please provide:
1. Temporal trends (if applicable)
2. Seasonal patterns
3. Growth or decline patterns
4. Trend significance and reliability
5. Future trend predictions
6. Recommendations based on trends

Focus on identifying patterns over time or sequences.
            """,
            
            "outliers": f"""
You are an outlier detection specialist. Analyze this dataset for outliers and anomalies:

{data_summary}

Please provide:
1. Outlier detection using multiple methods
2. Potential causes of outliers
3. Impact of outliers on analysis
4. Recommendations for handling outliers
5. Anomaly patterns and insights

Use statistical methods to identify and explain outliers.
            """,
            
            "comprehensive": f"""
You are a senior data scientist. Perform a comprehensive analysis of this dataset:

{data_summary}

Please provide a complete analysis including:
1. Executive Summary
2. Data Quality Assessment
3. Descriptive Statistics
4. Correlation Analysis
5. Trend Analysis
6. Outlier Detection
7. Key Insights and Patterns
8. Business Implications
9. Recommendations for Action
10. Further Analysis Suggestions

Provide a professional, comprehensive report suitable for business stakeholders.
            """
        }
        
        return prompts.get(analysis_type, prompts["comprehensive"])
    
    def _create_insights_prompt(self, df: pd.DataFrame, focus_area: str, business_context: str) -> str:
        """Create a prompt for generating business insights."""
        data_summary = self._prepare_data_summary(df)
        
        prompt = f"""
You are a business intelligence analyst. Generate actionable insights from this dataset:

{data_summary}

Focus Area: {focus_area}
Business Context: {business_context}

Please provide:
1. Key Business Insights
2. Performance Metrics Analysis
3. Opportunities and Threats
4. Strategic Recommendations
5. Action Items with Priority
6. Risk Assessment
7. Success Metrics
8. Implementation Timeline

Focus on practical, actionable insights that drive business value.
        """
        
        return prompt
    
    async def _generate_response(self, prompt: str, max_length: int = 2048) -> str:
        """Generate a response using the DeepSeek model."""
        if not self.is_loaded:
            await self.load_model()
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part (remove input prompt)
            generated_text = response[len(prompt):].strip()
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"Error in analysis: {str(e)}"
    
    async def analyze_data(self, df: pd.DataFrame, analysis_type: str = "comprehensive") -> str:
        """Analyze data using the DeepSeek model."""
        prompt = self._create_analysis_prompt(df, analysis_type)
        return await self._generate_response(prompt)
    
    async def generate_insights(self, df: pd.DataFrame, focus_area: str, business_context: str) -> str:
        """Generate business insights using the DeepSeek model."""
        prompt = self._create_insights_prompt(df, focus_area, business_context)
        return await self._generate_response(prompt)

# Global instance
_analyzer = None

async def get_analyzer() -> DeepSeekAnalyzer:
    """Get or create the global DeepSeek analyzer instance."""
    global _analyzer
    if _analyzer is None:
        _analyzer = DeepSeekAnalyzer()
    return _analyzer

async def analyze_with_deepseek(df: pd.DataFrame, analysis_type: str) -> str:
    """Analyze data using DeepSeek LLM."""
    analyzer = await get_analyzer()
    return await analyzer.analyze_data(df, analysis_type)

async def generate_insights_with_deepseek(df: pd.DataFrame, focus_area: str, business_context: str) -> str:
    """Generate insights using DeepSeek LLM."""
    analyzer = await get_analyzer()
    return await analyzer.generate_insights(df, focus_area, business_context)

# Alternative: Use a simpler approach for testing without loading the full model
class SimpleAnalyzer:
    """Simple analyzer for testing when DeepSeek model is not available."""
    
    @staticmethod
    def analyze_data(df: pd.DataFrame, analysis_type: str) -> str:
        """Perform basic analysis without LLM."""
        analysis = f"""
# Data Analysis Report - {analysis_type.title()}

## Dataset Overview
- **Shape**: {df.shape[0]} rows × {df.shape[1]} columns
- **Columns**: {', '.join(df.columns.tolist())}
- **Memory Usage**: {df.memory_usage(deep=True).sum() / 1024:.2f} KB

## Data Quality Assessment
- **Missing Values**: {df.isnull().sum().sum()} total missing values
- **Duplicate Rows**: {df.duplicated().sum()} duplicate rows
- **Data Types**: {dict(df.dtypes)}

## Basic Statistics
{df.describe().to_string()}

## Key Observations
1. Dataset contains {df.shape[0]} records with {df.shape[1]} features
2. Numeric columns: {list(df.select_dtypes(include=[np.number]).columns)}
3. Categorical columns: {list(df.select_dtypes(include=['object']).columns)}
4. Date columns: {list(df.select_dtypes(include=['datetime']).columns)}

## Recommendations
1. Review data quality and handle missing values
2. Analyze correlations between numeric variables
3. Create visualizations for key metrics
4. Consider feature engineering for better insights
        """
        
        return analysis
    
    @staticmethod
    def generate_insights(df: pd.DataFrame, focus_area: str, business_context: str) -> str:
        """Generate basic insights without LLM."""
        insights = f"""
# Business Insights Report

## Focus Area: {focus_area}
## Business Context: {business_context}

## Executive Summary
Based on analysis of {df.shape[0]} records, here are the key business insights:

## Key Metrics
- **Total Records**: {df.shape[0]}
- **Data Completeness**: {(1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100:.1f}%
- **Unique Values**: Average {df.nunique().mean():.1f} unique values per column

## Business Implications
1. **Data Quality**: {df.isnull().sum().sum()} missing values need attention
2. **Scale**: Dataset size suggests {'significant' if df.shape[0] > 1000 else 'moderate'} business activity
3. **Complexity**: {df.shape[1]} features indicate {'complex' if df.shape[1] > 10 else 'simple'} business processes

## Recommendations
1. **Immediate Actions**:
   - Address data quality issues
   - Validate business rules
   - Set up monitoring for key metrics

2. **Strategic Initiatives**:
   - Implement data governance
   - Develop automated reporting
   - Create dashboards for stakeholders

3. **Next Steps**:
   - Conduct detailed correlation analysis
   - Create predictive models
   - Establish KPIs and targets
        """
        
        return insights

# Fallback to simple analyzer if DeepSeek is not available
async def analyze_with_deepseek(df: pd.DataFrame, analysis_type: str) -> str:
    """Analyze data with fallback to simple analyzer."""
    try:
        analyzer = await get_analyzer()
        return await analyzer.analyze_data(df, analysis_type)
    except Exception as e:
        logger.warning(f"DeepSeek analysis failed, using simple analyzer: {str(e)}")
        return SimpleAnalyzer.analyze_data(df, analysis_type)

async def generate_insights_with_deepseek(df: pd.DataFrame, focus_area: str, business_context: str) -> str:
    """Generate insights with fallback to simple analyzer."""
    try:
        analyzer = await get_analyzer()
        return await analyzer.generate_insights(df, focus_area, business_context)
    except Exception as e:
        logger.warning(f"DeepSeek insights generation failed, using simple analyzer: {str(e)}")
        return SimpleAnalyzer.generate_insights(df, focus_area, business_context) 