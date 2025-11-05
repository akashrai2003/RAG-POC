"""
Startup script for the Asset RAG API server.
"""

import uvicorn
import sys
import os

# Add the src directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

if __name__ == "__main__":
    print("🚀 Starting Asset RAG API Server...")
    print("📡 API will be available at: http://localhost:8000")
    print("📚 Interactive docs at: http://localhost:8000/docs")
    print("🔧 Health check at: http://localhost:8000/health")
    print("\nPress Ctrl+C to stop the server\n")
    
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )