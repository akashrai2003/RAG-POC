"""
Parallel execution Agentic RAG system for asset data queries.
Fast, single-pass execution with parallel tool calls.
"""

from typing import Dict, List, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
import json
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

from config.settings import config, QueryConfig
from services.sql_service import SQLService
from services.rag_service import RAGService
from models.schemas import QueryRequest, QueryResponse


class ParallelAssetRAGAgent:
    """Fast agent with parallel SQL and RAG execution."""
    
    def __init__(self, sql_service: SQLService, rag_service: RAGService):
        self.sql_service = sql_service
        self.rag_service = rag_service
        
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=config.google_api_key,
            temperature=QueryConfig.AGENT_TEMPERATURE
        )
    
    async def query(self, query_request: QueryRequest) -> QueryResponse:
        """Process query with parallel execution."""
        start_time = time.time()
        
        try:
            print(f"\n{'#'*80}")
            print(f"🚀 STARTING PARALLEL AGENT QUERY")
            print(f"📝 User Query: {query_request.query}")
            print(f"{'#'*80}\n")
            
            # Step 1: LLM analyzes query and decides which tools to use
            print(f"{'='*80}")
            print(f"🧠 STEP 1: Query Analysis")
            print(f"{'='*80}")
            print(f"🤖 Asking OpenAI to analyze query and decide tools...")
            
            tool_decision = await self._analyze_query(query_request.query)
            
            use_sql = tool_decision.get('use_sql', False)
            use_rag = tool_decision.get('use_rag', False)
            sql_query_desc = tool_decision.get('sql_query', '')
            rag_question = tool_decision.get('rag_question', query_request.query)
            
            print(f"✅ Analysis complete:")
            print(f"   🔹 Use SQL: {use_sql}")
            print(f"   🔹 Use RAG: {use_rag}")
            if use_sql:
                print(f"   🔹 SQL Query: {sql_query_desc}")
            print(f"{'='*80}\n")
            
            # Step 2: Execute tools in parallel
            print(f"{'='*80}")
            print(f"⚡ STEP 2: Parallel Tool Execution")
            print(f"{'='*80}")
            
            sql_result = None
            rag_result = None
            tools_used = []
            
            # Execute in parallel using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = []
                
                if use_sql:
                    print(f"🔧 Launching SQL query in parallel...")
                    future_sql = executor.submit(self._execute_sql, sql_query_desc)
                    futures.append(('sql', future_sql))
                    tools_used.append('sql_query_tool')
                
                if use_rag:
                    print(f"🔧 Launching RAG query in parallel...")
                    future_rag = executor.submit(self._execute_rag, rag_question)
                    futures.append(('rag', future_rag))
                    tools_used.append('rag_query_tool')
                
                # Collect results
                for tool_type, future in futures:
                    result = future.result()
                    if tool_type == 'sql':
                        sql_result = result
                    else:
                        rag_result = result
            
            print(f"✅ All tools executed")
            print(f"{'='*80}\n")
            
            # Step 3: Generate final response
            print(f"{'='*80}")
            print(f"📝 STEP 3: Final Response Generation")
            print(f"{'='*80}")
            print(f"🤖 Generating final answer with collected results...")
            
            final_response = await self._generate_final_response(
                query_request.query,
                sql_result,
                rag_result
            )
            
            execution_time = time.time() - start_time
            
            # Determine query type
            query_type = "hybrid" if (use_sql and use_rag) else ("sql" if use_sql else "rag")
            
            print(f"✅ Final response generated")
            print(f"\n{'#'*80}")
            print(f"✅ QUERY COMPLETED")
            print(f"⏱️  Total Execution Time: {execution_time:.2f}s")
            print(f"🔧 Tools Used: {tools_used}")
            print(f"📊 Query Type: {query_type.upper()}")
            print(f"{'#'*80}\n")
            
            return QueryResponse(
                success=True,
                query_type=query_type,
                response=final_response,
                data=sql_result.get('data') if sql_result else None,
                sql_query=sql_result.get('sql_query') if sql_result else None,
                execution_time=execution_time,
                metadata={
                    "tools_used": tools_used,
                    "parallel_execution": True
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
    
    async def _analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze query and decide which tools to use."""
        
        analysis_prompt = f"""
You are an expert query analyzer for an asset tracking system. Analyze the user's query and decide which tools to use.

User Query: "{query}"

Available Tools:
1. SQL Tool - For numerical queries, filtering, counting, aggregations (voltage comparisons, counts, averages, etc.)
2. RAG Tool - For conceptual questions, explanations, definitions, context about processes

Analyze the query and respond with a JSON object:
{{
    "use_sql": true/false,
    "use_rag": true/false,
    "sql_query": "natural language description for SQL if use_sql is true, otherwise empty",
    "rag_question": "refined question for RAG if use_rag is true, otherwise original query"
}}

Guidelines:
- For pure numerical/filtering queries (voltage < 6, count assets, etc.): use_sql=true, use_rag=false
- For pure conceptual queries (what does X mean, explain Y): use_sql=false, use_rag=true  
- For queries needing both data and context: use_sql=true, use_rag=true
- Be conservative - only use RAG if contextual understanding is truly needed

Examples:
Query: "Show assets with battery voltage less than 6"
{{"use_sql": true, "use_rag": false, "sql_query": "assets with battery voltage less than 6", "rag_question": ""}}

Query: "What does In Network state mean?"
{{"use_sql": false, "use_rag": true, "sql_query": "", "rag_question": "What does In Network state mean?"}}

Query: "Show low voltage assets and explain why it's concerning"
{{"use_sql": true, "use_rag": true, "sql_query": "assets with low battery voltage", "rag_question": "why is low battery voltage concerning for asset tracking"}}

Respond with ONLY the JSON object, no other text.
"""
        
        response = self.llm.invoke(analysis_prompt)
        response_text = response.content.strip()
        
        # Extract JSON from response
        try:
            # Try to parse directly
            decision = json.loads(response_text)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            import re
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                decision = json.loads(json_match.group(1))
            else:
                # Fallback: assume it needs SQL for safety
                decision = {
                    "use_sql": True,
                    "use_rag": False,
                    "sql_query": query,
                    "rag_question": query
                }
        
        return decision
    
    def _execute_sql(self, query_description: str) -> Optional[Dict[str, Any]]:
        """Execute SQL query synchronously."""
        try:
            result = self.sql_service.execute_natural_language_query(query_description)
            # Return result whether it succeeded or failed
            return result
        except Exception as e:
            print(f"❌ SQL execution error: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'data': []
            }
    
    def _execute_rag(self, question: str) -> Optional[Dict[str, Any]]:
        """Execute RAG query synchronously."""
        try:
            result = self.rag_service.query(question)
            if result.get('success'):
                return result
            return None
        except Exception as e:
            print(f"❌ RAG execution error: {str(e)}")
            return None
    
    async def _generate_final_response(
        self,
        original_query: str,
        sql_result: Optional[Dict[str, Any]],
        rag_result: Optional[Dict[str, Any]]
    ) -> str:
        """Generate final response using both SQL and RAG results."""
        
        context_parts = []
        has_sql_data = False
        sql_error = None
        
        if sql_result:
            if sql_result.get('success'):
                sql_data = sql_result.get('data', [])
                sql_query = sql_result.get('sql_query', '')
                has_sql_data = True
                context_parts.append(f"""
SQL Query Results:
- SQL Query: {sql_query}
- Rows returned: {len(sql_data)}
- Data: {json.dumps(sql_data[:5], indent=2, default=str)}
{"(showing first 5 of " + str(len(sql_data)) + " rows)" if len(sql_data) > 5 else ""}
""")
            else:
                # SQL query failed
                sql_error = sql_result.get('error', 'Unknown SQL error')
                context_parts.append(f"""
SQL Query Failed:
- Error: {sql_error}
- This likely means the query used incorrect column names or syntax
""")
        
        if rag_result:
            rag_response = rag_result.get('response', '')
            context_parts.append(f"""
RAG Context:
{rag_response}
""")
        
        context = "\n\n".join(context_parts) if context_parts else "No data available"
        
        # Adjust prompt based on whether we have data or errors
        if sql_error:
            rag_context = ""
            if rag_result:
                rag_context = f"Additional Context:\n{rag_result.get('response', '')}"
            
            final_prompt = f"""
You are an expert assistant for asset tracking queries.

User's Question: "{original_query}"

IMPORTANT: The SQL query encountered an error, likely due to incorrect column names or database issues.

Error Information:
{sql_error}

{rag_context}

Provide a helpful response that:
1. Acknowledges that there was an issue retrieving the specific data
2. Explains the error in simple terms (e.g., "column not found" means the database doesn't have that field)
3. If RAG context is available, provide relevant general information
4. Suggest rephrasing the query or checking available data fields

Be honest about the limitation but helpful in guiding the user.

Answer:
"""
        elif has_sql_data:
            final_prompt = f"""
You are an expert assistant for asset tracking queries.

User's Question: "{original_query}"

Available Information:
{context}

Provide a clear, concise answer to the user's question based on the available information.
- Summarize the SQL data findings clearly
- State specific numbers and facts
- If RAG context is available, incorporate relevant explanations
- Be direct and factual

Answer:
"""
        else:
            final_prompt = f"""
You are an expert assistant for asset tracking queries.

User's Question: "{original_query}"

Available Information:
{context}

Provide a clear, concise answer to the user's question based on the available information.
- Be helpful and informative
- If limited data is available, state what you can provide
- Be honest about limitations

Answer:
"""
        
        response = self.llm.invoke(final_prompt)
        return response.content.strip()
