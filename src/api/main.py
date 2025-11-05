"""
FastAPI REST API for Asset RAG system.
Provides query endpoints to replace Streamlit interface.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import asyncio
import time
from datetime import datetime

# Import our services
from config.settings import config
from services.sql_service import SQLService
from services.rag_service import RAGService
from services.agent_service_parallel import ParallelAssetRAGAgent
from models.schemas import QueryRequest, QueryResponse


# Pydantic models for API
class QueryApiRequest(BaseModel):
    """API request model for queries."""
    query: str
    include_metadata: bool = True


class QueryApiResponse(BaseModel):
    """API response model for queries."""
    success: bool
    query_type: str
    response: str
    data: Optional[List[Dict[str, Any]]] = None
    sql_query: Optional[str] = None
    execution_time: float
    timestamp: str
    metadata: Optional[Dict[str, Any]] = None
    record_count: Optional[int] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    services: Dict[str, str]


# Initialize FastAPI app
app = FastAPI(
    title="Asset RAG API",
    description="REST API for querying asset data and documentation using natural language",
    version="1.0.0"
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global services (initialized on startup)
sql_service = None
rag_service = None
agent = None


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global sql_service, rag_service, agent
    
    print("\n" + "="*80)
    print("INITIALIZING API SERVICES")
    print("="*80)
    
    try:
        sql_service = SQLService()
        print("✅ SQL Service initialized")
        
        rag_service = RAGService()
        print("✅ RAG Service initialized")
        
        agent = ParallelAssetRAGAgent(sql_service, rag_service)
        print("✅ Parallel Agent initialized")
        
        print("="*80)
        print("🚀 API SERVER READY")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"❌ Failed to initialize services: {str(e)}")
        raise


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with basic API information."""
    return {
        "message": "Asset RAG API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "query": "/query",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        # Basic service checks
        sql_status = "healthy" if sql_service else "not_initialized"
        rag_status = "healthy" if rag_service else "not_initialized"
        agent_status = "healthy" if agent else "not_initialized"
        
        overall_status = "healthy" if all(
            status == "healthy" 
            for status in [sql_status, rag_status, agent_status]
        ) else "degraded"
        
        return HealthResponse(
            status=overall_status,
            timestamp=datetime.now().isoformat(),
            services={
                "sql_service": sql_status,
                "rag_service": rag_status,
                "agent_service": agent_status
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@app.post("/query", response_model=QueryApiResponse)
async def query_assets(request: QueryApiRequest):
    """
    Main query endpoint for natural language queries.
    
    Returns both the natural language response and structured data if available.
    """
    if not agent:
        raise HTTPException(status_code=503, detail="Services not initialized")
    
    try:
        print(f"\n{'='*80}")
        print(f"📝 API QUERY REQUEST")
        print(f"Query: {request.query}")
        print(f"{'='*80}")
        
        # Create internal query request
        query_request = QueryRequest(query=request.query)
        
        # Execute query using the parallel agent
        response = await agent.query(query_request)
        
        # Calculate record count if data is available
        record_count = len(response.data) if response.data else None
        
        # Prepare API response
        api_response = QueryApiResponse(
            success=response.success,
            query_type=response.query_type,
            response=response.response,
            data=response.data,
            sql_query=response.sql_query,
            execution_time=response.execution_time,
            timestamp=datetime.now().isoformat(),
            metadata=response.metadata if request.include_metadata else None,
            record_count=record_count
        )
        
        print(f"✅ Query completed successfully")
        print(f"⏱️  Execution time: {response.execution_time:.2f}s")
        print(f"📊 Records returned: {record_count}")
        print(f"🔧 Query type: {response.query_type}")
        print(f"{'='*80}\n")
        
        return api_response
    
    except Exception as e:
        print(f"❌ Query failed: {str(e)}")
        print(f"{'='*80}\n")
        
        # Return error response
        return QueryApiResponse(
            success=False,
            query_type="error",
            response=f"Error processing query: {str(e)}",
            execution_time=0.0,
            timestamp=datetime.now().isoformat(),
            metadata={"error": str(e)} if request.include_metadata else None
        )


@app.get("/examples")
async def get_query_examples():
    """Get example queries for different use cases."""
    return {
        "data_queries": [
            "Show assets with battery voltage less than 6",
            "Count assets by state",
            "List all assets with Cardinal Tags",
            "Show me assets shipped after 2024-01-01",
            "What's the average battery voltage?"
        ],
        "documentation_queries": [
            "What is a Cardinal Tag?",
            "Explain the 'In Transit' state",
            "What does FRIG mean?",
            "Describe the AT3 Pilot product",
            "What actions are typically needed?"
        ],
        "hybrid_queries": [
            "Show low voltage assets and explain why it's concerning",
            "List assets in transit and explain the process",
            "Show Cardinal Tag assets and explain their purpose"
        ]
    }


@app.get("/stats")
async def get_system_stats():
    """Get basic system statistics."""
    if not sql_service:
        raise HTTPException(status_code=503, detail="SQL service not initialized")
    
    try:
        # Get basic counts
        asset_count_result = sql_service.execute_natural_language_query("count all assets")
        
        stats = {
            "total_assets": 0,
            "timestamp": datetime.now().isoformat()
        }
        
        if asset_count_result.get('success') and asset_count_result.get('data'):
            # Extract count from the result
            data = asset_count_result['data']
            if data and len(data) > 0:
                # Get the first numeric value from the first row
                first_row = data[0]
                for value in first_row.values():
                    if isinstance(value, (int, float)):
                        stats["total_assets"] = int(value)
                        break
        
        return stats
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)