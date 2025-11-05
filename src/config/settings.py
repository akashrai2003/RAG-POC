"""
Configuration management for RAG-POC application.
"""

import os
from typing import List
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class AppConfig(BaseSettings):
    """Application configuration settings."""
    
    # App Information
    app_name: str = Field(default="Asset RAG POC")
    app_version: str = Field(default="1.0.0")
    debug: bool = Field(default=True)

    # Google API
    # google_api_key: str = Field(default="")

    # # OpenAI API
    openai_api_key: str = Field(default="")
    
    # Database Configuration
    sqlite_db_path: str = Field(default="data/db/assets.db")
    chroma_db_path: str = Field(default="data/db/chroma_db")
    
    # File Upload Configuration
    max_file_size_mb: int = Field(default=50)
    allowed_file_types: List[str] = Field(default_factory=lambda: ["xlsx", "xls", "csv"])
    
    # Query Configuration
    max_query_length: int = Field(default=500)
    default_similarity_threshold: float = Field(default=0.7)
    
    # Logging Configuration
    log_level: str = Field(default="INFO")
    log_file: str = Field(default="logs/app.log")
    
    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "ignore"
    }


class DatabaseConfig:
    """Database configuration constants."""
    
    # SQLite configuration
    SQLITE_TIMEOUT = 30
    SQLITE_CHECK_SAME_THREAD = False
    
    # ChromaDB configuration
    CHROMA_COLLECTION_NAME = "asset_documents"
    CHROMA_DISTANCE_FUNCTION = "cosine"
    
    # Embedding model
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class QueryConfig:
    """Query processing configuration for LangGraph agents."""
    
    # Agent configuration
    MAX_ITERATIONS = 2  # Reduced to prevent unnecessary calls
    AGENT_VERBOSE = True
    
    # SQL Tool configuration
    SQL_TOOL_DESCRIPTION = """
    Use this tool when you need to perform mathematical operations, filtering, counting, 
    or any numerical analysis on the asset data. This includes queries about:
    - Battery voltage comparisons (>, <, =, between values)
    - Counting assets by various criteria
    - Date/time filtering and comparisons
    - Aggregations (sum, average, min, max)
    - Sorting and ordering data
    - Statistical analysis
    
    The tool expects a natural language description of what data you want to retrieve,
    and it will generate and execute the appropriate SQL query.
    """
    
    # RAG Tool configuration  
    RAG_TOOL_DESCRIPTION = """
    Use this tool when you need to answer questions about:
    - Conceptual understanding of asset data and business processes
    - Explanations of what different asset states mean
    - Product descriptions and specifications
    - General information about asset management
    - Context about locations, accounts, or business processes
    - Any question that requires understanding meaning rather than computation
    
    This tool searches through documentation and provides contextual answers.
    """
    
    # Temperature settings for different operations
    AGENT_TEMPERATURE = 0.3        # Agent decision making
    SQL_GENERATION_TEMPERATURE = 0.1  # SQL generation (needs precision)
    RAG_RESPONSE_TEMPERATURE = 0.4    # Natural language responses


# Global configuration instance
config = AppConfig()