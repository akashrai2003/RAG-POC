"""
Simple test client for the Asset RAG API.
Demonstrates how to interact with the REST API endpoints.
"""

import requests
import json
from typing import Dict, Any


class AssetRAGClient:
    """Client for interacting with the Asset RAG API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health status."""
        try:
            response = requests.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}
    
    def query(self, query: str, include_metadata: bool = True) -> Dict[str, Any]:
        """Send a natural language query to the API."""
        try:
            payload = {
                "query": query,
                "include_metadata": include_metadata
            }
            
            response = requests.post(
                f"{self.base_url}/query",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json()
        
        except requests.RequestException as e:
            return {"error": str(e)}
    
    def get_examples(self) -> Dict[str, Any]:
        """Get example queries."""
        try:
            response = requests.get(f"{self.base_url}/examples")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        try:
            response = requests.get(f"{self.base_url}/stats")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}


def main():
    """Test the API with some example queries."""
    client = AssetRAGClient()
    
    print("🧪 Testing Asset RAG API")
    print("=" * 50)
    
    # Test health check
    print("\n1. Health Check:")
    health = client.health_check()
    print(json.dumps(health, indent=2))
    
    # Test stats
    print("\n2. System Stats:")
    stats = client.get_stats()
    print(json.dumps(stats, indent=2))
    
    # Test example queries
    print("\n3. Available Examples:")
    examples = client.get_examples()
    if "error" not in examples:
        print("Data Queries:")
        for query in examples.get("data_queries", [])[:3]:
            print(f"  - {query}")
        print("\nDocumentation Queries:")
        for query in examples.get("documentation_queries", [])[:3]:
            print(f"  - {query}")
    
    # Test a sample query
    print("\n4. Sample Query Test:")
    sample_query = "Show assets with battery voltage less than 6"
    print(f"Query: {sample_query}")
    
    result = client.query(sample_query)
    
    if "error" not in result:
        print(f"\nSuccess: {result['success']}")
        print(f"Query Type: {result['query_type']}")
        print(f"Execution Time: {result['execution_time']:.2f}s")
        print(f"Record Count: {result.get('record_count', 'N/A')}")
        print(f"\nResponse: {result['response'][:200]}...")
        
        if result.get('data'):
            print(f"\nFirst few data records:")
            for i, record in enumerate(result['data'][:3]):
                print(f"  Record {i+1}: {record}")
    else:
        print(f"Error: {result['error']}")


if __name__ == "__main__":
    main()