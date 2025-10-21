"""
Smart Agentic RAG system with tool calling and intelligent routing.
Uses LLM tool calling to decide if SQL is needed, while summarizing RAG context.
SUPER-OPTIMIZED: 1 LLM call for pure RAG, 2 for data queries.
"""

from typing import Dict, List, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
import json
import time

from config.settings import config, QueryConfig
from services.sql_service import SQLService
from services.rag_service import RAGService
from models.schemas import QueryRequest, QueryResponse


@tool
def use_sql_query(reason: str) -> str:
    """
    Call this tool if the user's query requires data from Salesforce/database.
    
    Args:
        reason: Brief explanation of why SQL is needed
        
    Returns:
        Confirmation that SQL will be used
    """
    return f"SQL query will be executed: {reason}"


class SmartAssetRAGAgent:
    """
    Super-optimized agent using tool calling for intelligent routing.
    
    Flow for Pure RAG Query (e.g., "What is FRIG?"):
    1. LLM analyzes query + RAG context → Generates answer (no tool call)
    → TOTAL: 1 LLM call
    
    Flow for Data Query (e.g., "Show assets with Cardinal Tags"):
    1. LLM analyzes query + RAG context → Calls use_sql tool + summarizes RAG
    2. Execute SQL with RAG summary → Generate final response
    → TOTAL: 2 LLM calls
    
    Key Innovation:
    - Single LLM call produces BOTH tool call decision AND RAG summary
    - RAG summary is SQL-aware (understands schema, helps with term mapping)
    - Pure documentation queries answered in 1 call!
    """
    
    def __init__(self, sql_service: SQLService, rag_service: RAGService):
        self.sql_service = sql_service
        self.rag_service = rag_service
        
        # Initialize LLM with tool calling capability
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=config.openai_api_key,
            temperature=QueryConfig.AGENT_TEMPERATURE
        ).bind_tools([use_sql_query])
    
    async def query(self, query_request: QueryRequest) -> QueryResponse:
        """Process query with intelligent routing via tool calling."""
        start_time = time.time()
        
        try:
            print(f"\n{'#'*80}")
            print(f"🚀 STARTING SMART AGENT QUERY")
            print(f"📝 User Query: {query_request.query}")
            print(f"{'#'*80}\n")
            
            # ==========================================
            # STEP 1: Retrieve RAG Context + Analyze Query
            # ==========================================
            print(f"{'='*80}")
            print(f"🧠 STEP 1: Intelligent Analysis (with tool calling)")
            print(f"{'='*80}")
            
            # Get RAG context from documents
            print(f"📚 Retrieving RAG context from documentation...")
            rag_result = self.rag_service.query(query_request.query, n_results=5)
            
            rag_context = ""
            rag_sources = []
            
            if rag_result.get('success'):
                rag_context = rag_result.get('response', '')
                rag_sources = rag_result.get('sources', [])
                print(f"✅ Retrieved RAG context: {len(rag_sources)} sources, {len(rag_context)} chars")
            
            # Get schema info from SQL service for SQL-aware RAG summary
            schema_info = self.sql_service._get_schema_info()
            
            # Build comprehensive prompt for LLM
            system_prompt = f"""You are an expert assistant for SMART Logistics asset tracking system.

You have access to:
1. Documentation context (RAG) - definitions, processes, business terms
2. Salesforce database - live asset data, metrics, counts

Database Schema (for reference):
{schema_info}

Your job:
1. **Analyze the user's query**
2. **Decide if SQL/data query is needed**:
   - If user wants specific data, counts, lists, reports → Call use_sql_query tool
   - If user asks about definitions, processes, concepts → Don't call tool, answer directly
3. **Summarize RAG context** in a way that:
   - Explains business terms mentioned in the query
   - Maps terms to database columns (e.g., "Cardinal Tag" → Cardinal_Tag__c)
   - Provides context that would help SQL generation

Documentation Context Available:
{rag_context if rag_context else "No relevant documentation found."}

Think step by step:
- Does the query need live data from database? → use_sql_query tool
- Can I answer from documentation alone? → Direct response
"""

            user_message = f"""User Query: "{query_request.query}"

Instructions:
1. If this query needs data from Salesforce (counts, lists, specific assets, metrics), call the use_sql_query tool with a brief reason
2. Provide a helpful summary of the documentation context, especially:
   - Any business terms in the query and their database column mappings  
   - **CRITICAL**: If the documentation defines HOW something is calculated (formulas, conditions, business rules), extract that logic explicitly
   - Example: If asking about "Rogue" assets, explain the FORMULA/LOGIC that defines rogue (not just "it's a status field")
   - Relevant definitions, processes, or business rules
   - Context that would help generate accurate SQL queries with proper calculations

If you called use_sql_query tool, your summary will be used to help generate the SQL query.
If you didn't call the tool, provide a complete answer to the user's question using the documentation.

When extracting business logic, use format:
- Term: [Business Term]
- Logic: [Calculation formula or condition]
- Fields: [Database columns involved]"""

            print(f"🤖 LLM Call #1: Analyzing query with tool calling...")
            
            # Call LLM with tool binding
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message)
            ]
            
            response = self.llm.invoke(messages)
            
            # Check if tool was called
            tool_calls = response.tool_calls if hasattr(response, 'tool_calls') else []
            needs_sql = len(tool_calls) > 0
            
            # Get the LLM's text response (RAG summary or direct answer)
            llm_response = response.content if response.content else ""
            
            # If tool was called but no text response, create a basic RAG summary
            if needs_sql and not llm_response and rag_context:
                llm_response = f"Documentation context: {rag_context[:500]}"
                print(f"   ℹ️  Using RAG context directly (LLM provided no summary)")
            
            print(f"\n📊 Analysis Result:")
            print(f"   🔧 SQL Needed: {needs_sql}")
            if needs_sql and tool_calls:
                print(f"   📝 Reason: {tool_calls[0]['args'].get('reason', 'Data query required')}")
            print(f"   📚 RAG Summary Length: {len(llm_response)} chars")
            print(f"{'='*80}\n")
            
            # ==========================================
            # STEP 2 (Conditional): Execute SQL if needed
            # ==========================================
            if needs_sql:
                print(f"{'='*80}")
                print(f"🔧 STEP 2: Generate & Execute SAQL Query")
                print(f"{'='*80}")
                print(f"🤖 LLM Call #2: Generating SAQL with RAG summary...")
                
                # Use the LLM's summary as RAG context for SQL generation
                saql_result = self.sql_service.execute_natural_language_query(
                    query_request.query,
                    rag_context=llm_response  # Use LLM's SQL-aware summary!
                )
                
                if saql_result.get('success'):
                    print(f"✅ SAQL executed: {len(saql_result.get('data', []))} rows")
                else:
                    print(f"⚠️  SAQL failed: {saql_result.get('error')}")
                
                print(f"{'='*80}\n")
                
                # Generate final response combining data + documentation
                print(f"{'='*80}")
                print(f"📝 STEP 3: Generate Final Response")
                print(f"{'='*80}")
                
                final_response = await self._generate_final_response(
                    query_request.query,
                    saql_result,
                    llm_response,  # Reuse RAG summary
                    rag_sources
                )
                
                execution_time = time.time() - start_time
                
                print(f"✅ Final response generated ({len(final_response)} chars)")
                print(f"📝 Response Preview: {final_response[:200]}...")
                print(f"\n{'#'*80}")
                print(f"✅ QUERY COMPLETED (Data Query)")
                print(f"⏱️  Total Time: {execution_time:.2f}s")
                print(f"💰 LLM Calls: 2 (Analysis + Final Response)")
                print(f"{'#'*80}\n")
                
                return QueryResponse(
                    success=True,
                    query_type="data",
                    response=final_response,
                    data=saql_result.get('data'),
                    sql_query=saql_result.get('saql_query'),
                    execution_time=execution_time,
                    metadata={
                        "llm_calls": 2,
                        "rag_sources": len(rag_sources),
                        "optimization": "smart_routing_with_tools"
                    }
                )
            
            else:
                # Pure RAG query - we already have the answer!
                execution_time = time.time() - start_time
                
                print(f"{'#'*80}")
                print(f"✅ QUERY COMPLETED (Pure RAG)")
                print(f"⏱️  Total Time: {execution_time:.2f}s")
                print(f"💰 LLM Calls: 1 (Single analysis call!)")
                print(f"📚 RAG Sources: {len(rag_sources)}")
                print(f"{'#'*80}\n")
                
                return QueryResponse(
                    success=True,
                    query_type="rag",
                    response=llm_response,  # Use the direct answer
                    execution_time=execution_time,
                    metadata={
                        "llm_calls": 1,
                        "rag_sources": len(rag_sources),
                        "optimization": "single_call_rag"
                    }
                )
        
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"\n{'#'*80}")
            print(f"❌ QUERY FAILED")
            print(f"🚨 Error: {str(e)}")
            print(f"{'#'*80}\n")
            
            return QueryResponse(
                success=False,
                query_type="error",
                response=f"Error: {str(e)}",
                execution_time=execution_time,
                metadata={"error": str(e)}
            )
    
    async def _generate_final_response(
        self,
        user_query: str,
        saql_result: Dict[str, Any],
        rag_summary: str,
        rag_sources: List[Dict[str, Any]]
    ) -> str:
        """Generate final response combining SAQL results and RAG context."""
        
        print(f"🔄 Generating final response...")
        print(f"   Query: {user_query}")
        print(f"   RAG Summary: {len(rag_summary)} chars")
        print(f"   Data Rows: {len(saql_result.get('data', [])) if saql_result.get('success') else 0}")
        
        # Build data context
        data_context = ""
        if saql_result and saql_result.get('success'):
            data = saql_result.get('data', [])
            data_context = f"""
SALESFORCE DATA:
- Rows returned: {len(data)}
- Data: {json.dumps(data[:5], indent=2, default=str)}
{f"(Showing first 5 of {len(data)} rows)" if len(data) > 5 else ""}
"""
        
        # Build source citations
        sources_text = ""
        if rag_sources:
            sources_text = "\n\nSources:\n" + "\n".join([
                f"- {s['file_name']} (Page {s['page']})" 
                for s in rag_sources[:3]
            ])
        
        prompt = f"""You are an expert assistant for SMART Logistics.

User Query: "{user_query}"

{data_context}

DOCUMENTATION CONTEXT (already analyzed):
{rag_summary}
{sources_text}

Provide a comprehensive answer that:
1. Presents the data findings clearly
2. Explains concepts using the documentation context
3. Combines both for a complete, conversational answer

Answer:"""
        
        print(f"🤖 Calling LLM to generate final response...")
        # Use a fresh LLM instance WITHOUT tool binding for final response
        response_llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=config.openai_api_key,
            temperature=QueryConfig.AGENT_TEMPERATURE
        )
        response = response_llm.invoke(prompt)
        final_text = response.content.strip()
        print(f"✅ LLM returned response: {len(final_text)} chars")
        return final_text
    
    def query_sync(self, query_request: QueryRequest) -> QueryResponse:
        """Synchronous wrapper."""
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        
        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, self.query(query_request))
                return future.result()
        else:
            return asyncio.run(self.query(query_request))
