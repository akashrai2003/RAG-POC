"""
Optimized Agentic RAG system with early context retrieval.
Retrieves RAG context once and uses it for both SAQL generation and final response.
COST-OPTIMIZED: Reduces LLM calls from 3 to 2.
"""

from typing import Dict, List, Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
import json
import time

from config.settings import config, QueryConfig
from services.sql_service import SQLService
from services.rag_service import RAGService
from models.schemas import QueryRequest, QueryResponse


class OptimizedAssetRAGAgent:
    """
    Cost-optimized agent that retrieves RAG context early and uses it throughout.
    
    Flow:
    1. User query comes in
    2. Retrieve RAG context from PDFs (helps understand business terms)
    3. Generate SAQL query using RAG context (better column targeting)
    4. Execute SAQL query
    5. Generate final response using both SAQL results and RAG context (already retrieved)
    
    Benefits:
    - Only 2 LLM calls instead of 3 (cost savings ~33%)
    - Better SAQL queries (understands business terminology)
    - Richer final responses (combines data + documentation)
    """
    
    def __init__(self, sql_service: SQLService, rag_service: RAGService):
        self.sql_service = sql_service
        self.rag_service = rag_service
        
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=config.google_api_key,
            temperature=QueryConfig.AGENT_TEMPERATURE
        )
    
    async def query(self, query_request: QueryRequest) -> QueryResponse:
        """Process query with early RAG context retrieval."""
        start_time = time.time()
        
        try:
            print(f"\n{'#'*80}")
            print(f"🚀 STARTING OPTIMIZED AGENT QUERY")
            print(f"📝 User Query: {query_request.query}")
            print(f"{'#'*80}\n")
            
            # ==========================================
            # STEP 1: Retrieve RAG Context Early
            # ==========================================
            print(f"{'='*80}")
            print(f"📚 STEP 1: Retrieve RAG Context (Early)")
            print(f"{'='*80}")
            print(f"🔍 Searching documentation for relevant context...")
            
            rag_context = None
            rag_sources = []
            
            try:
                # Get RAG context with more results for better coverage
                rag_result = self.rag_service.query(query_request.query, n_results=5)
                
                if rag_result.get('success'):
                    rag_context = rag_result.get('response', '')
                    rag_sources = rag_result.get('sources', [])
                    
                    print(f"✅ Retrieved RAG context:")
                    print(f"   📄 Sources found: {len(rag_sources)}")
                    print(f"   📊 Context length: {len(rag_context)} characters")
                    print(f"   🎯 Confidence: {rag_result.get('confidence', 0):.1%}")
                    
                    # Show source documents
                    if rag_sources:
                        print(f"\n   📚 Source documents:")
                        for i, source in enumerate(rag_sources[:3], 1):
                            print(f"      {i}. {source['file_name']} (Page {source['page']}) - {source['relevance_score']:.1%}")
                else:
                    print(f"⚠️  RAG context retrieval had no results")
                    
            except Exception as e:
                print(f"⚠️  RAG context retrieval failed: {str(e)}")
                rag_context = None
            
            print(f"{'='*80}\n")
            
            # ==========================================
            # STEP 2: Generate SAQL Query with RAG Context
            # ==========================================
            print(f"{'='*80}")
            print(f"🔧 STEP 2: Generate SAQL Query (with RAG context)")
            print(f"{'='*80}")
            print(f"🤖 LLM Call #1: Generating SAQL query...")
            print(f"   💡 Using RAG context to understand business terms")
            
            saql_result = await self._generate_and_execute_saql(
                query_request.query,
                rag_context
            )
            
            if saql_result.get('success'):
                print(f"✅ SAQL query executed successfully")
                print(f"   📊 Rows returned: {len(saql_result.get('data', []))}")
                if saql_result.get('saql_query'):
                    print(f"   🔍 SAQL: {saql_result['saql_query'][:100]}...")
            else:
                print(f"⚠️  SAQL query failed: {saql_result.get('error', 'Unknown error')}")
            
            print(f"{'='*80}\n")
            
            # ==========================================
            # STEP 3: Generate Final Response
            # ==========================================
            print(f"{'='*80}")
            print(f"📝 STEP 3: Generate Final Response")
            print(f"{'='*80}")
            print(f"🤖 LLM Call #2: Generating final answer...")
            print(f"   📊 Using SAQL results + RAG context (already retrieved)")
            
            final_response = await self._generate_final_response(
                query_request.query,
                saql_result,
                rag_context,
                rag_sources
            )
            
            execution_time = time.time() - start_time
            
            print(f"✅ Final response generated")
            print(f"\n{'#'*80}")
            print(f"✅ QUERY COMPLETED")
            print(f"⏱️  Total Execution Time: {execution_time:.2f}s")
            print(f"💰 LLM Calls Used: 2 (optimized)")
            print(f"📚 RAG Sources: {len(rag_sources)}")
            print(f"{'#'*80}\n")
            
            return QueryResponse(
                success=True,
                query_type="hybrid",  # Always uses both SAQL and RAG
                response=final_response,
                data=saql_result.get('data') if saql_result else None,
                sql_query=saql_result.get('saql_query') if saql_result else None,
                execution_time=execution_time,
                metadata={
                    "llm_calls": 2,
                    "rag_sources": len(rag_sources),
                    "optimization": "early_rag_retrieval"
                }
            )
        
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"\n{'#'*80}")
            print(f"❌ QUERY FAILED")
            print(f"⏱️  Execution Time: {execution_time:.2f}s")
            print(f"🚨 Error: {str(e)}")
            print(f"{'#'*80}\n")
            
            return QueryResponse(
                success=False,
                query_type="error",
                response=f"Error processing query: {str(e)}",
                execution_time=execution_time,
                metadata={"error": str(e)}
            )
    
    async def _generate_and_execute_saql(
        self, 
        user_query: str,
        rag_context: Optional[str]
    ) -> Dict[str, Any]:
        """
        Generate SAQL query using RAG context for better understanding of business terms.
        
        This is crucial because:
        1. User might use business terms not in the prompt (e.g., "FRIG", "Cardinal Tag")
        2. RAG context helps map these terms to correct database columns
        3. Results in more accurate SAQL queries
        
        This method delegates to SQL service's comprehensive SAQL generation logic.
        """
        
        try:
            # Use SQL service's execute_natural_language_query with RAG context
            # This uses the SQL service's well-tested prompt with all SAQL rules
            result = self.sql_service.execute_natural_language_query(
                user_query, 
                rag_context=rag_context
            )
            
            if result.get('success'):
                print(f"   ✅ Query successful: {len(result.get('data', []))} rows returned")
            else:
                print(f"   ⚠️  SAQL query failed: {result.get('error', 'Unknown error')}")
            
            return result
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'data': []
            }
    
    async def _generate_final_response(
        self,
        user_query: str,
        saql_result: Dict[str, Any],
        rag_context: Optional[str],
        rag_sources: List[Dict[str, Any]]
    ) -> str:
        """
        Generate final response using both SAQL results and RAG context.
        
        This provides a rich answer that:
        1. Presents the data findings
        2. Explains concepts using documentation
        3. Cites sources when appropriate
        """
        
        # Build context from SAQL results
        saql_context = ""
        if saql_result and saql_result.get('success'):
            data = saql_result.get('data', [])
            saql_query = saql_result.get('saql_query', '')
            
            saql_context = f"""
SALESFORCE DATA RESULTS:
- Query executed: {saql_query}
- Rows returned: {len(data)}
- Data preview: {json.dumps(data[:5], indent=2, default=str)}
{f"(Showing first 5 of {len(data)} rows)" if len(data) > 5 else ""}
"""
        elif saql_result:
            saql_context = f"""
SALESFORCE QUERY ISSUE:
- Error: {saql_result.get('error', 'Query failed')}
- No data available from Salesforce
"""
        
        # Build context from RAG
        rag_context_formatted = ""
        if rag_context:
            rag_context_formatted = f"""
DOCUMENTATION CONTEXT:
{rag_context}

Sources:
{chr(10).join([f"- {s['file_name']} (Page {s['page']})" for s in rag_sources[:3]])}
"""
        
        # Generate final response
        final_prompt = f"""
You are an expert assistant for SMART Logistics asset tracking system.

User's Question: "{user_query}"

Available Information:

{saql_context}

{rag_context_formatted}

Your Task:
Provide a comprehensive answer that:
1. **Presents the data findings** clearly (if SAQL data available)
   - State specific numbers, counts, and facts
   - Highlight key insights from the data
   
2. **Explains concepts** using the documentation context
   - Define business terms mentioned
   - Provide context about processes or states
   
3. **Combines both** for a complete answer
   - Data tells "what" (current state, numbers)
   - Documentation explains "why" and "what it means"

4. **Cites sources** when using documentation
   - Reference document names when explaining concepts

5. **Be conversational** but professional
   - Address the user's question directly
   - Use clear, business-friendly language

If SAQL data is not available, focus on providing helpful context from documentation.

Answer:
"""
        
        response = self.llm.invoke(final_prompt)
        return response.content.strip()
    
    def query_sync(self, query_request: QueryRequest) -> QueryResponse:
        """Synchronous wrapper for async query method."""
        import asyncio
        
        # Check if we're already in an event loop
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        
        if loop and loop.is_running():
            # We're in an async context, create a new event loop in a thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, self.query(query_request))
                return future.result()
        else:
            # No event loop, safe to use asyncio.run
            return asyncio.run(self.query(query_request))
