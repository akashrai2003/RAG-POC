"""
LangGraph-based Agentic RAG system for asset data queries.
"""

from typing import TypedDict, List, Optional, Any, Dict, Type
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_core.pydantic_v1 import Field as PydanticField
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import sqlite3
import json
import time

from config.settings import config, QueryConfig
from services.sql_service import SQLService
from services.rag_service import RAGService
from models.schemas import QueryRequest, QueryResponse


class AgentState(TypedDict):
    """State of the RAG agent."""
    messages: List[BaseMessage]
    query: str
    query_result: Optional[str]
    data_result: Optional[List[Dict[str, Any]]]
    tool_calls: List[str]
    iteration_count: int
    final_answer: Optional[str]
    execution_time: float


class SQLQueryTool(BaseTool):
    """Tool for executing SQL queries on asset data."""
    
    name: str = "sql_query_tool"
    description: str = QueryConfig.SQL_TOOL_DESCRIPTION
    sql_service: Any = PydanticField(default=None, exclude=True)
    
    def __init__(self, sql_service: SQLService, **kwargs):
        super().__init__(**kwargs)
        self.sql_service = sql_service
    
    def _run(self, query_description: str) -> str:
        """Execute SQL query based on natural language description."""
        try:
            result = self.sql_service.execute_natural_language_query(query_description)
            
            if result.get('success'):
                data = result.get('data', [])
                if data:
                    # Format the results for the agent
                    formatted_result = {
                        'sql_query': result.get('sql_query'),
                        'row_count': len(data),
                        'data': data[:10] if len(data) > 10 else data,  # Limit to first 10 rows
                        'truncated': len(data) > 10
                    }
                    return json.dumps(formatted_result, indent=2, default=str)
                else:
                    return "Query executed successfully but returned no results."
            else:
                return f"SQL query failed: {result.get('error', 'Unknown error')}"
        
        except Exception as e:
            return f"Error executing SQL query: {str(e)}"
    
    async def _arun(self, query_description: str) -> str:
        """Async version of _run."""
        return self._run(query_description)


class RAGQueryTool(BaseTool):
    """Tool for semantic search and RAG-based responses."""
    
    name: str = "rag_query_tool"
    description: str = QueryConfig.RAG_TOOL_DESCRIPTION
    rag_service: Any = PydanticField(default=None, exclude=True)
    
    def __init__(self, rag_service: RAGService, **kwargs):
        super().__init__(**kwargs)
        self.rag_service = rag_service
    
    def _run(self, question: str) -> str:
        """Get RAG-based response for semantic questions."""
        try:
            result = self.rag_service.query(question)
            
            if result.get('success'):
                response = result.get('response', '')
                sources = result.get('sources', [])
                
                formatted_result = {
                    'answer': response,
                    'sources_used': len(sources),
                    'confidence': result.get('confidence', 0.0)
                }
                
                if sources:
                    formatted_result['source_snippets'] = sources[:3]  # First 3 sources
                
                return json.dumps(formatted_result, indent=2)
            else:
                return f"RAG query failed: {result.get('error', 'Unknown error')}"
        
        except Exception as e:
            return f"Error executing RAG query: {str(e)}"
    
    async def _arun(self, question: str) -> str:
        """Async version of _run."""
        return self._run(question)


class AssetRAGAgent:
    """LangGraph-based agent for asset data queries."""
    
    def __init__(self, sql_service: SQLService, rag_service: RAGService):
        self.sql_service = sql_service
        self.rag_service = rag_service
        
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=config.google_api_key,
            temperature=QueryConfig.AGENT_TEMPERATURE
        )
        
        # Initialize tools
        self.tools = [
            SQLQueryTool(sql_service),
            RAGQueryTool(rag_service)
        ]
        
        # Create tool node for LangGraph
        self.tool_node = ToolNode(self.tools)
        
        # Bind tools to LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("tools", self.tool_node)
        workflow.add_node("final_response", self._final_response_node)
        
        # Set entry point
        workflow.set_entry_point("agent")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "continue": "tools",
                "end": "final_response"
            }
        )
        
        # Add edge from tools back to agent
        workflow.add_edge("tools", "agent")
        
        # Add edge from final_response to END
        workflow.add_edge("final_response", END)
        
        return workflow.compile()
    
    def _agent_node(self, state: AgentState) -> AgentState:
        """Agent decision-making node."""
        iteration = state["iteration_count"] + 1
        print(f"\n{'='*80}")
        print(f"🤖 AGENT ITERATION #{iteration}")
        print(f"{'='*80}")
        
        messages = state["messages"]
        
        # Create system message with context about available tools
        system_message = self._create_system_message()
        
        # Prepare messages for LLM
        llm_messages = [system_message] + messages
        
        # Get response from LLM
        print(f"🧠 Agent analyzing query and deciding on action...")
        response = self.llm_with_tools.invoke(llm_messages)
        
        # Add response to messages
        state["messages"].append(response)
        state["iteration_count"] += 1
        
        # Track tool calls
        if hasattr(response, 'tool_calls') and response.tool_calls:
            for tool_call in response.tool_calls:
                state["tool_calls"].append(tool_call["name"])
                print(f"🔧 Agent decided to use tool: {tool_call['name']}")
                print(f"   Input: {tool_call.get('args', {})}")
        else:
            print(f"✅ Agent has enough information - preparing final response")
        
        return state
    
    def _extract_tool_results(self, state: AgentState) -> AgentState:
        """Extract results from tool messages."""
        # Find the last tool message
        for message in reversed(state["messages"]):
            if isinstance(message, ToolMessage):
                result_content = message.content
                
                # Try to parse as JSON
                try:
                    result_data = json.loads(result_content)
                    if "data" in result_data:
                        state["data_result"] = result_data.get("data", [])
                    state["query_result"] = result_content
                except:
                    state["query_result"] = result_content
                break
        
        return state
    
    def _final_response_node(self, state: AgentState) -> AgentState:
        """Generate final response to user."""
        # Extract tool results if any
        state = self._extract_tool_results(state)
        # Create a prompt for final response generation
        final_prompt = f"""
        Based on the analysis and tool results, provide a clear, comprehensive answer to the user's question: "{state['query']}"
        
        Tool results: {state.get('query_result', 'No tool results')}
        
        Provide a natural language response that directly answers the user's question.
        If you retrieved specific data, summarize the key findings.
        If you used SQL, explain what the data shows.
        If you used RAG, provide the contextual information requested.
        """
        
        final_response = self.llm.invoke([HumanMessage(content=final_prompt)])
        state["final_answer"] = final_response.content
        
        return state
    
    def _should_continue(self, state: AgentState) -> str:
        """Decide whether to continue with tool use or provide final answer."""
        last_message = state["messages"][-1]
        
        # Check if we've hit max iterations
        if state["iteration_count"] >= QueryConfig.MAX_ITERATIONS:
            return "end"
        
        # Check if the last message has tool calls
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "continue"
        
        return "end"
    
    def _create_system_message(self) -> BaseMessage:
        """Create system message with context about available tools and data."""
        system_content = f"""
        You are an intelligent assistant for querying asset tracking data. You have access to two powerful tools:

        1. **SQL Query Tool**: Use this for mathematical operations, filtering, counting, aggregations, and any data analysis involving numbers, dates, or precise filtering.

        2. **RAG Query Tool**: Use this for conceptual questions, explanations, and when you need contextual understanding about asset management processes.

        **Asset Data Schema:**
        The database contains asset records with these key fields:
        - Asset_ID__c: Unique identifier
        - Battery_Voltage__c: Numerical voltage readings
        - Account_Name__c: Account/customer names
        - Product_Name__c: Product types (e.g., "AT3 Pilot", "AT5 - Bracket")
        - State_of_Pallet__c: Current state (e.g., "In Network", "In Transit")
        - Date_Shipped__c: Shipping dates
        - Last_Connected__c: Last connection timestamps
        - Current_Location_Name__c: Current locations
        - Action_Needed__c: Required actions
        - Power_Reset_Occurred__c: Power reset status

        **IMPORTANT Guidelines:**
        - Choose ONLY ONE tool that best matches the query type
        - For numerical/filtering queries (like "voltage less than 6"), use ONLY SQL tool
        - For conceptual questions, use ONLY RAG tool
        - DO NOT call multiple tools for simple queries
        - After getting results from one tool, provide your final answer immediately
        - Be concise and direct with your tool selection

        Analyze the user's question carefully and choose the single most appropriate tool.
        """
        
        return HumanMessage(content=system_content)
    
    async def query(self, query_request: QueryRequest) -> QueryResponse:
        """Process a query using the agent."""
        start_time = time.time()
        
        try:
            print(f"\n{'#'*80}")
            print(f"🚀 STARTING AGENT QUERY")
            print(f"📝 User Query: {query_request.query}")
            print(f"{'#'*80}\n")
            
            # Initialize state
            initial_state = AgentState(
                messages=[HumanMessage(content=query_request.query)],
                query=query_request.query,
                query_result=None,
                data_result=None,
                tool_calls=[],
                iteration_count=0,
                final_answer=None,
                execution_time=0.0
            )
            
            # Run the graph
            final_state = await self.graph.ainvoke(initial_state)
            
            execution_time = time.time() - start_time
            
            # Determine query type based on tools used
            query_type = "hybrid"
            unique_tools = list(set(final_state["tool_calls"]))
            if final_state["tool_calls"]:
                if "sql_query_tool" in unique_tools and "rag_query_tool" not in unique_tools:
                    query_type = "sql"
                elif "rag_query_tool" in unique_tools and "sql_query_tool" not in unique_tools:
                    query_type = "rag"
            
            print(f"\n{'#'*80}")
            print(f"✅ QUERY COMPLETED")
            print(f"⏱️  Total Execution Time: {execution_time:.2f}s")
            print(f"🔧 Tools Used: {unique_tools}")
            print(f"🔄 Total Iterations: {final_state['iteration_count']}")
            print(f"📊 Query Type: {query_type.upper()}")
            print(f"{'#'*80}\n")
            
            return QueryResponse(
                success=True,
                query_type=query_type,
                response=final_state["final_answer"] or "No response generated",
                data=final_state.get("data_result"),
                sql_query=self._extract_sql_query(final_state.get("query_result")),
                execution_time=execution_time,
                metadata={
                    "tools_used": unique_tools,
                    "iterations": final_state["iteration_count"]
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
    
    def _extract_sql_query(self, query_result: Optional[str]) -> Optional[str]:
        """Extract SQL query from tool result if available."""
        if not query_result:
            return None
        
        try:
            result_data = json.loads(query_result)
            return result_data.get("sql_query")
        except:
            return None