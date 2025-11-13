"""
Parallel execution Agentic RAG system for asset data queries.
Fast, single-pass execution with parallel tool calls.
"""

from typing import Dict, List, Any, Optional
from langchain_openai import ChatOpenAI
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
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            openai_api_key=config.openai_api_key,
            temperature=QueryConfig.AGENT_TEMPERATURE
        )
    
    async def query(self, query_request: QueryRequest) -> QueryResponse:
        """Process query with parallel execution."""
        start_time = time.time()
        
        # Initialize token tracking
        total_input_tokens = 0
        total_output_tokens = 0
        
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
            
            tool_decision_result = await self._analyze_query(query_request.query)
            tool_decision = tool_decision_result['decision']
            total_input_tokens += tool_decision_result['input_tokens']
            total_output_tokens += tool_decision_result['output_tokens']
            
            use_sql = tool_decision.get('use_sql', False)
            use_rag = tool_decision.get('use_rag', False)
            sql_query_desc = tool_decision.get('sql_query', '')
            rag_question = tool_decision.get('rag_question', query_request.query)
            
            # IMPORTANT: If SQL is needed, always include RAG for business context
            if use_sql and not use_rag:
                use_rag = True
                print(f"ℹ️  Auto-enabling RAG to provide business context for SQL query")
            
            print(f"✅ Analysis complete:")
            print(f"   🔹 Use SQL: {use_sql}")
            print(f"   🔹 Use RAG: {use_rag} {'(auto-enabled for SQL context)' if use_sql else ''}")
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
                        # Extract token usage from SQL service
                        if result and 'token_usage' in result:
                            sql_tokens = result['token_usage']
                            total_input_tokens += sql_tokens.get('input_tokens', 0)
                            total_output_tokens += sql_tokens.get('output_tokens', 0)
                            print(f"   💰 SQL generation used: {sql_tokens.get('input_tokens', 0)} input, {sql_tokens.get('output_tokens', 0)} output tokens")
                    else:
                        rag_result = result
                        # Extract token usage from RAG service
                        if result and 'token_usage' in result:
                            rag_tokens = result['token_usage']
                            total_input_tokens += rag_tokens.get('input_tokens', 0)
                            total_output_tokens += rag_tokens.get('output_tokens', 0)
                            print(f"   💰 RAG generation used: {rag_tokens.get('input_tokens', 0)} input, {rag_tokens.get('output_tokens', 0)} output tokens")
            
            print(f"✅ All tools executed")
            
            # Print RAG context that will be used for SQL query generation
            if rag_result and rag_result.get('success'):
                print(f"\n{'~'*80}")
                print(f"📚 RAG CONTEXT FOR SQL GENERATION:")
                print(f"{'~'*80}")
                rag_response = rag_result.get('response', '')
                print(f"{rag_response}")
                print(f"{'~'*80}")
            
            print(f"{'='*80}\n")
            
            # Step 3: Generate final response
            print(f"{'='*80}")
            print(f"📝 STEP 3: Final Response Generation")
            print(f"{'='*80}")
            print(f"🤖 Generating final answer with collected results...")
            
            final_response_result = await self._generate_final_response(
                query_request.query,
                sql_result,
                rag_result
            )
            final_response = final_response_result['response']
            total_input_tokens += final_response_result['input_tokens']
            total_output_tokens += final_response_result['output_tokens']
            
            # Calculate cost (gpt-4o-mini pricing)
            input_cost = (total_input_tokens / 1_000_000) * 0.150
            output_cost = (total_output_tokens / 1_000_000) * 0.600
            total_cost = input_cost + output_cost
            
            execution_time = time.time() - start_time
            
            # Determine query type
            query_type = "hybrid" if (use_sql and use_rag) else ("sql" if use_sql else "rag")
            
            print(f"✅ Final response generated")
            print(f"\n{'#'*80}")
            print(f"✅ QUERY COMPLETED")
            print(f"⏱️  Total Execution Time: {execution_time:.2f}s")
            print(f"🔧 Tools Used: {tools_used}")
            print(f"📊 Query Type: {query_type.upper()}")
            print(f"💰 Token Usage:")
            print(f"   🔹 Input tokens: {total_input_tokens:,}")
            print(f"   🔹 Output tokens: {total_output_tokens:,}")
            print(f"   🔹 Total tokens: {total_input_tokens + total_output_tokens:,}")
            print(f"   🔹 Cost: ${total_cost:.6f} (Input: ${input_cost:.6f}, Output: ${output_cost:.6f})")
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
                    "parallel_execution": True,
                    "token_usage": {
                        "input_tokens": total_input_tokens,
                        "output_tokens": total_output_tokens,
                        "total_tokens": total_input_tokens + total_output_tokens
                    },
                    "cost": {
                        "input_cost": input_cost,
                        "output_cost": output_cost,
                        "total_cost": total_cost,
                        "currency": "USD"
                    }
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
        """Analyze query and decide which tools to use. Returns decision and token usage."""
        
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
        print(response_text)
        # Track token usage
        input_tokens = response.response_metadata.get('token_usage', {}).get('prompt_tokens', 0)
        output_tokens = response.response_metadata.get('token_usage', {}).get('completion_tokens', 0)
        
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
        
        return {
            'decision': decision,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens
        }
    
    def _execute_sql(self, query_description: str) -> Optional[Dict[str, Any]]:
        """Execute SQL query synchronously."""
        try:
            result = self.sql_service.execute_natural_language_query(query_description)
            if result and isinstance(result, dict):
                # copy saql_query -> sql_query if missing
                if 'saql_query' in result and 'sql_query' not in result:
                    result['sql_query'] = result.get('saql_query')

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
    ) -> Dict[str, Any]:
        """Generate final response using both SQL and RAG results. Returns response and token usage."""
        
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
        
        # Track token usage
        input_tokens = response.response_metadata.get('token_usage', {}).get('prompt_tokens', 0)
        output_tokens = response.response_metadata.get('token_usage', {}).get('completion_tokens', 0)
        
        return {
            'response': response.content.strip(),
            'input_tokens': input_tokens,
            'output_tokens': output_tokens
        }
