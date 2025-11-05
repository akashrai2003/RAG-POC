"""
FastAPI backend for RAG-POC with conversation history and query processing.
"""

from fastapi import FastAPI, HTTPException, Depends, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any
import time
import logging
from datetime import datetime
import asyncio

# Import our services and models
from config.settings import config
from services.sql_service import SQLService
from services.rag_service import RAGService
from services.conversation_service import ConversationService
from services.agent_service_parallel import ParallelAssetRAGAgent
from models.schemas import (
    QueryRequest, QueryResponse,
    QueryWithConversationRequest, QueryWithConversationResponse,
    CreateConversationRequest, CreateMessageRequest,
    Conversation, ConversationWithMessages, Message,
    ConversationListResponse, ConversationStatsResponse,
    MessageRole
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global service instances
services = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    # Startup
    logger.info("🚀 Starting RAG-POC FastAPI Backend...")
    
    try:
        # Initialize services
        logger.info("📊 Initializing SQL Service...")
        sql_service = SQLService()
        
        logger.info("🔍 Initializing RAG Service...")
        rag_service = RAGService()
        
        logger.info("💬 Initializing Conversation Service...")
        conversation_service = ConversationService()
        
        logger.info("🤖 Initializing Agent Service...")
        agent_service = ParallelAssetRAGAgent(sql_service, rag_service)
        
        # Store services globally
        services.update({
            'sql': sql_service,
            'rag': rag_service,
            'conversation': conversation_service,
            'agent': agent_service
        })
        
        logger.info("✅ All services initialized successfully!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize services: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down RAG-POC FastAPI Backend...")


# Create FastAPI app
app = FastAPI(
    title="RAG-POC API",
    description="REST API for Asset RAG system with conversation history",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependency to get services
def get_sql_service() -> SQLService:
    return services['sql']

def get_rag_service() -> RAGService:
    return services['rag']

def get_conversation_service() -> ConversationService:
    return services['conversation']

def get_agent_service() -> ParallelAssetRAGAgent:
    return services['agent']


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "RAG-POC FastAPI Backend",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "timestamp": datetime.utcnow().isoformat()
    }


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check if services are available
        sql_service = get_sql_service()
        rag_service = get_rag_service()
        conversation_service = get_conversation_service()
        
        # Get some basic stats
        rag_stats = rag_service.get_collection_stats()
        conv_stats = conversation_service.get_conversation_stats()
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "services": {
                "sql_service": "available",
                "rag_service": "available", 
                "conversation_service": "available",
                "agent_service": "available"
            },
            "stats": {
                "rag_documents": rag_stats.get('total_chunks', 0),
                "total_conversations": conv_stats.total_conversations,
                "total_messages": conv_stats.total_messages
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


# Query endpoints
@app.post("/api/v1/query", response_model=QueryResponse)
async def process_query(
    request: QueryRequest,
    agent_service: ParallelAssetRAGAgent = Depends(get_agent_service)
):
    """Process a standalone query without conversation context."""
    try:
        logger.info(f"Processing standalone query: {request.query}")
        result = await agent_service.query(request)
        return result
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@app.post("/api/v1/query-with-conversation", response_model=QueryWithConversationResponse)
async def process_query_with_conversation(
    request: QueryWithConversationRequest,
    agent_service: ParallelAssetRAGAgent = Depends(get_agent_service),
    conversation_service: ConversationService = Depends(get_conversation_service)
):
    """Process a query within conversation context."""
    try:
        logger.info(f"Processing query with conversation: {request.query}")
        
        # Handle conversation creation or retrieval
        conversation_id = request.conversation_id
        
        if not conversation_id and request.create_conversation:
            # Create new conversation
            conv_request = CreateConversationRequest(
                title=request.conversation_title or f"Chat - {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}",
                user_id=request.user_id,
                initial_message=request.query
            )
            conversation = conversation_service.create_conversation(conv_request)
            conversation_id = conversation.id
        elif conversation_id:
            # Add user message to existing conversation
            msg_request = CreateMessageRequest(
                conversation_id=conversation_id,
                role=MessageRole.USER,
                content=request.query
            )
            conversation_service.add_message(msg_request)
        else:
            raise HTTPException(status_code=400, detail="Either provide conversation_id or set create_conversation=true")
        
        # Process the query
        query_request = QueryRequest(
            query=request.query,
            context=request.context
        )
        
        result = await agent_service.query(query_request)
        
        # Add assistant response to conversation
        assistant_msg = CreateMessageRequest(
            conversation_id=conversation_id,
            role=MessageRole.ASSISTANT,
            content=result.response,
            metadata={
                "query_type": result.query_type,
                "execution_time": result.execution_time,
                "sql_query": result.sql_query,
                "tools_used": result.metadata.get("tools_used", [])
            }
        )
        
        assistant_message = conversation_service.add_message(assistant_msg)
        
        return QueryWithConversationResponse(
            success=result.success,
            query_type=result.query_type,
            response=result.response,
            conversation_id=conversation_id,
            message_id=assistant_message.id,
            data=result.data,
            sql_query=result.sql_query,
            execution_time=result.execution_time,
            metadata=result.metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query with conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


# Conversation management endpoints
@app.post("/api/v1/conversations", response_model=Conversation)
async def create_conversation(
    request: CreateConversationRequest,
    conversation_service: ConversationService = Depends(get_conversation_service)
):
    """Create a new conversation."""
    try:
        logger.info(f"Creating new conversation: {request.title}")
        conversation = conversation_service.create_conversation(request)
        return conversation
    except Exception as e:
        logger.error(f"Error creating conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create conversation: {str(e)}")


@app.get("/api/v1/conversations", response_model=ConversationListResponse)
async def list_conversations(
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=100, description="Items per page"),
    conversation_service: ConversationService = Depends(get_conversation_service)
):
    """List conversations with pagination."""
    try:
        logger.info(f"Listing conversations: page={page}, page_size={page_size}, user_id={user_id}")
        result = conversation_service.list_conversations(
            user_id=user_id,
            page=page,
            page_size=page_size
        )
        return result
    except Exception as e:
        logger.error(f"Error listing conversations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list conversations: {str(e)}")


@app.get("/api/v1/conversations/{conversation_id}", response_model=ConversationWithMessages)
async def get_conversation(
    conversation_id: str,
    include_messages: bool = Query(True, description="Include messages in response"),
    conversation_service: ConversationService = Depends(get_conversation_service)
):
    """Get a specific conversation by ID."""
    try:
        logger.info(f"Getting conversation: {conversation_id}")
        conversation = conversation_service.get_conversation(
            conversation_id=conversation_id,
            include_messages=include_messages
        )
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        return conversation
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get conversation: {str(e)}")


@app.put("/api/v1/conversations/{conversation_id}/title")
async def update_conversation_title(
    conversation_id: str,
    title: str,
    conversation_service: ConversationService = Depends(get_conversation_service)
):
    """Update a conversation's title."""
    try:
        logger.info(f"Updating conversation title: {conversation_id}")
        success = conversation_service.update_conversation_title(conversation_id, title)
        
        if not success:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        return {"success": True, "message": "Title updated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating conversation title: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update title: {str(e)}")


@app.delete("/api/v1/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    conversation_service: ConversationService = Depends(get_conversation_service)
):
    """Delete a conversation and all its messages."""
    try:
        logger.info(f"Deleting conversation: {conversation_id}")
        success = conversation_service.delete_conversation(conversation_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        return {"success": True, "message": "Conversation deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete conversation: {str(e)}")


# Message endpoints
@app.post("/api/v1/conversations/{conversation_id}/messages", response_model=Message)
async def add_message(
    conversation_id: str,
    content: str,
    role: str = MessageRole.USER,
    metadata: Optional[Dict[str, Any]] = None,
    conversation_service: ConversationService = Depends(get_conversation_service)
):
    """Add a message to a conversation."""
    try:
        logger.info(f"Adding message to conversation: {conversation_id}")
        
        request = CreateMessageRequest(
            conversation_id=conversation_id,
            role=role,
            content=content,
            metadata=metadata or {}
        )
        
        message = conversation_service.add_message(request)
        return message
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error adding message: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to add message: {str(e)}")


# Search and statistics endpoints
@app.get("/api/v1/conversations/search")
async def search_conversations(
    q: str = Query(..., description="Search query"),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    limit: int = Query(10, ge=1, le=50, description="Maximum results"),
    conversation_service: ConversationService = Depends(get_conversation_service)
):
    """Search conversations by content or title."""
    try:
        logger.info(f"Searching conversations: {q}")
        results = conversation_service.search_conversations(
            query=q,
            user_id=user_id,
            limit=limit
        )
        return {"results": results, "count": len(results)}
    except Exception as e:
        logger.error(f"Error searching conversations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.get("/api/v1/conversations/stats", response_model=ConversationStatsResponse)
async def get_conversation_stats(
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    conversation_service: ConversationService = Depends(get_conversation_service)
):
    """Get conversation statistics."""
    try:
        logger.info("Getting conversation statistics")
        stats = conversation_service.get_conversation_stats(user_id=user_id)
        return stats
    except Exception as e:
        logger.error(f"Error getting conversation stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


# RAG service endpoints
@app.get("/api/v1/rag/stats")
async def get_rag_stats(
    rag_service: RAGService = Depends(get_rag_service)
):
    """Get RAG service statistics."""
    try:
        stats = rag_service.get_collection_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting RAG stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get RAG stats: {str(e)}")


@app.get("/api/v1/rag/documents")
async def list_rag_documents(
    rag_service: RAGService = Depends(get_rag_service)
):
    """List loaded RAG documents."""
    try:
        documents = rag_service.list_loaded_documents()
        return {"documents": documents, "count": len(documents)}
    except Exception as e:
        logger.error(f"Error listing RAG documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")


# Admin endpoints
@app.post("/api/v1/admin/cleanup")
async def cleanup_old_conversations(
    background_tasks: BackgroundTasks,
    days_old: int = Query(30, ge=1, description="Delete conversations older than this many days"),
    conversation_service: ConversationService = Depends(get_conversation_service)
):
    """Cleanup old conversations (admin endpoint)."""
    try:
        logger.info(f"Cleaning up conversations older than {days_old} days")
        
        def cleanup_task():
            deleted_count = conversation_service.cleanup_old_conversations(days_old)
            logger.info(f"Cleanup completed: {deleted_count} conversations deleted")
        
        background_tasks.add_task(cleanup_task)
        
        return {
            "success": True,
            "message": f"Cleanup task started for conversations older than {days_old} days"
        }
    except Exception as e:
        logger.error(f"Error starting cleanup: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start cleanup: {str(e)}")


@app.get("/api/v1/admin/database-info")
async def get_database_info(
    conversation_service: ConversationService = Depends(get_conversation_service)
):
    """Get database information (admin endpoint)."""
    try:
        info = conversation_service.get_database_info()
        return info
    except Exception as e:
        logger.error(f"Error getting database info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get database info: {str(e)}")


# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc), "type": "ValueError"}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": "InternalError"}
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )