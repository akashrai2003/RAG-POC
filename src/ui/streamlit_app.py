"""
Streamlit web interface for the Asset RAG POC.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import time
import asyncio
from typing import Optional, Dict, Any

# Import our services
from config.settings import config
from services.sql_service import SQLService
from services.rag_service import RAGService
from services.data_service import DataProcessingService
from src.services.agent_service_old import AssetRAGAgent
from models.schemas import QueryRequest


class AssetRAGInterface:
    """Streamlit interface for the Asset RAG system."""
    
    def __init__(self):
        self.setup_page_config()
        self.initialize_services()
    
    def setup_page_config(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="Asset RAG POC",
            page_icon="🏷️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        st.markdown("""
        <style>
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        .query-result {
            background-color: #e8f4fd;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #0066cc;
        }
        .error-message {
            background-color: #ffeaa7;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #e17055;
        }
        .success-message {
            background-color: #d5f4e6;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #00b894;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def initialize_services(self):
        """Initialize all services."""
        if 'services_initialized' not in st.session_state:
            try:
                # Initialize services
                st.session_state.sql_service = SQLService()
                st.session_state.rag_service = RAGService()
                st.session_state.data_service = DataProcessingService(
                    st.session_state.sql_service,
                    st.session_state.rag_service
                )
                st.session_state.agent = AssetRAGAgent(
                    st.session_state.sql_service,
                    st.session_state.rag_service
                )
                
                # Add business context to RAG
                st.session_state.rag_service.add_business_context()
                
                st.session_state.services_initialized = True
                
            except Exception as e:
                st.error(f"Failed to initialize services: {str(e)}")
                st.stop()
    
    def run(self):
        """Main application runner."""
        # Header
        st.title("🏷️ Asset RAG POC")
        st.markdown("*Intelligent asset data querying with hybrid RAG + SQL approach*")
        
        # Sidebar
        self.render_sidebar()
        
        # Main content
        tab1, tab2, tab3, tab4 = st.tabs(["💬 Query Assets", "📊 Dashboard", "📁 Data Management", "⚙️ System Status"])
        
        with tab1:
            self.render_query_interface()
        
        with tab2:
            self.render_dashboard()
        
        with tab3:
            self.render_data_management()
        
        with tab4:
            self.render_system_status()
    
    def render_sidebar(self):
        """Render the sidebar with configuration options."""
        st.sidebar.header("🔧 Configuration")
        
        # API Key status
        api_key_status = "✅ Configured" if config.openai_api_key else "❌ Missing"
        st.sidebar.markdown(f"**Google API Key:** {api_key_status}")
        
        # Query settings
        st.sidebar.subheader("Query Settings")
        
        # Temperature setting (stored in session state)
        temperature = st.sidebar.slider(
            "Response Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.1,
            help="Higher values make responses more creative, lower values more focused"
        )
        
        # Max results
        max_results = st.sidebar.number_input(
            "Max Results",
            min_value=1,
            max_value=100,
            value=10,
            help="Maximum number of results to return"
        )
        
        # Store in session state
        st.session_state.temperature = temperature
        st.session_state.max_results = max_results
        
        # Quick stats
        st.sidebar.subheader("📊 Quick Stats")
        self.render_quick_stats()
        
        # Sample queries
        st.sidebar.subheader("💡 Sample Queries")
        sample_queries = [
            "Show assets with battery voltage less than 6",
            "Count assets by state",
            "What is an AT3 Pilot?",
            "List assets that need RMA",
            "Average battery voltage by product type",
            "Explain asset states"
        ]
        
        for query in sample_queries:
            if st.sidebar.button(query, key=f"sample_{hash(query)}"):
                st.session_state.sample_query = query
    
    def render_quick_stats(self):
        """Render quick statistics in sidebar."""
        try:
            stats = st.session_state.sql_service.get_database_stats()
            if 'error' not in stats:
                st.sidebar.metric("Total Assets", stats.get('total_records', 0))
                
                # Battery voltage stats
                voltage_stats = stats.get('voltage_stats', {})
                if voltage_stats.get('count', 0) > 0:
                    avg_voltage = voltage_stats.get('avg_voltage', 0)
                    st.sidebar.metric("Avg Battery Voltage", f"{avg_voltage:.2f}V")
        except:
            st.sidebar.info("Load data to see statistics")
    
    def render_query_interface(self):
        """Render the main query interface."""
        st.header("💬 Ask Questions About Your Assets")
        
        # Query input
        col1, col2 = st.columns([4, 1])
        
        with col1:
            # Check if there's a sample query to use
            default_query = st.session_state.get('sample_query', '')
            if default_query:
                query_text = st.text_input(
                    "Enter your question:",
                    value=default_query,
                    placeholder="e.g., Show me assets with battery voltage less than 6"
                )
                # Clear the sample query
                if 'sample_query' in st.session_state:
                    del st.session_state.sample_query
            else:
                query_text = st.text_input(
                    "Enter your question:",
                    placeholder="e.g., Show me assets with battery voltage less than 6"
                )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)  # Add some space
            query_button = st.button("🔍 Query", type="primary", use_container_width=True)
        
        # Process query
        if query_button and query_text:
            self.process_query(query_text)
        
        # Display query history
        if 'query_history' in st.session_state and st.session_state.query_history:
            st.subheader("📝 Recent Queries")
            
            for i, (query, response, timestamp) in enumerate(reversed(st.session_state.query_history[-5:])):
                with st.expander(f"{timestamp}: {query[:50]}..."):
                    st.markdown(f"**Query:** {query}")
                    st.markdown(f"**Response:** {response.get('response', 'No response')}")
                    
                    if response.get('data'):
                        st.markdown("**Data Results:**")
                        df = pd.DataFrame(response['data'])
                        st.dataframe(df, use_container_width=True)
    
    def process_query(self, query_text: str):
        """Process a user query using the agent."""
        with st.spinner("🤖 Processing your query..."):
            try:
                # Create query request
                query_request = QueryRequest(query=query_text)
                
                # Process with agent (need to handle async)
                response = asyncio.run(st.session_state.agent.query(query_request))
                
                # Display results
                self.display_query_results(query_text, response)
                
                # Store in history
                if 'query_history' not in st.session_state:
                    st.session_state.query_history = []
                
                st.session_state.query_history.append((
                    query_text,
                    response.dict(),
                    datetime.now().strftime("%H:%M:%S")
                ))
                
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
    
    def display_query_results(self, query: str, response):
        """Display query results."""
        st.subheader("🔍 Query Results")
        
        # Success/failure indicator
        if response.success:
            st.success(f"✅ Query processed successfully ({response.execution_time:.2f}s)")
        else:
            st.error("❌ Query failed")
            return
        
        # Query type indicator
        query_type_colors = {
            'sql': '🔢',
            'rag': '🧠',
            'hybrid': '🔄'
        }
        query_icon = query_type_colors.get(response.query_type, '❓')
        st.info(f"{query_icon} Query Type: **{response.query_type.upper()}**")
        
        # Response
        st.markdown("### 💬 Answer")
        st.markdown(f'<div class="query-result">{response.response}</div>', unsafe_allow_html=True)
        
        # Show SQL query if available
        if response.sql_query:
            st.markdown("### 🔍 Generated SQL Query")
            st.code(response.sql_query, language='sql')
        
        # Show data results if available
        if response.data:
            st.markdown("### 📊 Data Results")
            df = pd.DataFrame(response.data)
            
            # Display data table
            st.dataframe(df, use_container_width=True)
            
            # Download button
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            # Simple visualization for numeric data
            self.create_simple_visualization(df)
        
        # Metadata
        if response.metadata:
            with st.expander("🔍 Query Details"):
                st.json(response.metadata)
    
    def create_simple_visualization(self, df: pd.DataFrame):
        """Create simple visualizations for data results."""
        try:
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            
            if not numeric_cols:
                return
            
            st.markdown("### 📈 Quick Visualization")
            
            # If battery voltage is present, show a histogram
            if 'battery_voltage' in df.columns:
                fig = px.histogram(
                    df, 
                    x='battery_voltage',
                    title="Battery Voltage Distribution",
                    nbins=20
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # If we have state information, show a pie chart
            elif 'state_of_pallet' in df.columns:
                state_counts = df['state_of_pallet'].value_counts()
                fig = px.pie(
                    values=state_counts.values,
                    names=state_counts.index,
                    title="Asset State Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            # Don't fail on visualization errors
            pass
    
    def render_dashboard(self):
        """Render the analytics dashboard."""
        st.header("📊 Asset Analytics Dashboard")
        
        try:
            stats = st.session_state.sql_service.get_database_stats()
            
            if 'error' in stats:
                st.warning("No data available. Please upload asset data first.")
                return
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Assets", stats.get('total_records', 0))
            
            with col2:
                voltage_stats = stats.get('voltage_stats', {})
                avg_voltage = voltage_stats.get('avg_voltage', 0)
                st.metric("Avg Battery Voltage", f"{avg_voltage:.2f}V")
            
            with col3:
                state_dist = stats.get('state_distribution', {})
                active_assets = state_dist.get('In Network', 0)
                st.metric("Active Assets", active_assets)
            
            with col4:
                low_battery_count = len([v for v in stats.get('voltage_stats', {}).values() if isinstance(v, (int, float)) and v < 6])
                st.metric("Low Battery Alerts", low_battery_count)
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                # State distribution pie chart
                if state_dist:
                    fig = px.pie(
                        values=list(state_dist.values()),
                        names=list(state_dist.keys()),
                        title="Asset State Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Product distribution bar chart
                product_dist = stats.get('product_distribution', {})
                if product_dist:
                    fig = px.bar(
                        x=list(product_dist.keys()),
                        y=list(product_dist.values()),
                        title="Assets by Product Type"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Battery voltage analysis
            st.subheader("🔋 Battery Voltage Analysis")
            
            # Get detailed battery data
            battery_query = "SELECT asset_id, battery_voltage, product_name, state_of_pallet FROM assets WHERE battery_voltage IS NOT NULL"
            battery_result = st.session_state.sql_service._execute_sql_query(battery_query)
            
            if battery_result.get('success') and battery_result.get('data'):
                battery_df = pd.DataFrame(battery_result['data'])
                
                # Voltage histogram
                fig = px.histogram(
                    battery_df,
                    x='battery_voltage',
                    color='product_name',
                    title="Battery Voltage Distribution by Product",
                    nbins=20
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Low battery assets
                low_battery_df = battery_df[battery_df['battery_voltage'] < 6]
                if not low_battery_df.empty:
                    st.subheader("⚠️ Low Battery Assets (< 6V)")
                    st.dataframe(low_battery_df, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error loading dashboard: {str(e)}")
    
    def render_data_management(self):
        """Render data management interface."""
        st.header("📁 Data Management")
        
        # File upload
        st.subheader("📤 Upload Asset Data")
        
        uploaded_file = st.file_uploader(
            "Choose an Excel or CSV file",
            type=['xlsx', 'xls', 'csv'],
            help="Upload asset data in Excel or CSV format"
        )
        
        if uploaded_file is not None:
            if st.button("Process File", type="primary"):
                self.process_uploaded_file(uploaded_file)
        
        # Sample data
        st.subheader("🎲 Sample Data")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Load Sample Data"):
                self.load_sample_data()
        
        with col2:
            if st.button("Clear All Data", type="secondary"):
                self.clear_all_data()
        
        # Current data preview
        st.subheader("👀 Current Data Preview")
        try:
            preview_query = "SELECT * FROM assets LIMIT 10"
            preview_result = st.session_state.sql_service._execute_sql_query(preview_query)
            
            if preview_result.get('success') and preview_result.get('data'):
                preview_df = pd.DataFrame(preview_result['data'])
                st.dataframe(preview_df, use_container_width=True)
            else:
                st.info("No data available")
        except Exception as e:
            st.warning("Unable to load data preview")
    
    def process_uploaded_file(self, uploaded_file):
        """Process an uploaded file."""
        with st.spinner("Processing uploaded file..."):
            try:
                result = st.session_state.data_service.process_uploaded_file(
                    uploaded_file,
                    uploaded_file.name
                )
                
                if result.success:
                    st.success(f"✅ Successfully processed {result.records_processed} records!")
                    
                    # Show processing details
                    with st.expander("📋 Processing Details"):
                        st.json({
                            'filename': result.filename,
                            'records_processed': result.records_processed,
                            'processing_time': f"{result.processing_time:.2f}s",
                            'schema_validation': result.schema_validation
                        })
                else:
                    st.error("❌ File processing failed")
                    for error in result.errors:
                        st.error(f"• {error}")
                        
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    def load_sample_data(self):
        """Load sample data for testing."""
        with st.spinner("Loading sample data..."):
            try:
                result = st.session_state.data_service.create_sample_data()
                
                if result.success:
                    st.success(f"✅ Loaded {result.records_processed} sample records!")
                else:
                    st.error("❌ Failed to load sample data")
                    for error in result.errors:
                        st.error(f"• {error}")
                        
            except Exception as e:
                st.error(f"Error loading sample data: {str(e)}")
    
    def clear_all_data(self):
        """Clear all data from databases."""
        if st.button("⚠️ Confirm Clear All Data", type="secondary"):
            with st.spinner("Clearing all data..."):
                try:
                    # Clear SQL database
                    sql_result = st.session_state.sql_service.clear_database()
                    
                    # Clear vector database
                    rag_result = st.session_state.rag_service.clear_collection()
                    
                    if sql_result.get('success') and rag_result.get('success'):
                        st.success("✅ All data cleared successfully!")
                    else:
                        st.error("❌ Failed to clear some data")
                        
                except Exception as e:
                    st.error(f"Error clearing data: {str(e)}")
    
    def render_system_status(self):
        """Render system status and configuration."""
        st.header("⚙️ System Status")
        
        # Service status
        st.subheader("🟢 Service Status")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # SQL Service
            try:
                sql_stats = st.session_state.sql_service.get_database_stats()
                sql_status = "🟢 Online" if 'error' not in sql_stats else "🔴 Error"
                st.metric("SQL Service", sql_status)
                
                if 'error' not in sql_stats:
                    st.metric("Total Records", sql_stats.get('total_records', 0))
            except:
                st.metric("SQL Service", "🔴 Offline")
        
        with col2:
            # RAG Service
            try:
                rag_stats = st.session_state.rag_service.get_collection_stats()
                rag_status = "🟢 Online" if 'error' not in rag_stats else "🔴 Error"
                st.metric("RAG Service", rag_status)
                
                if 'error' not in rag_stats:
                    st.metric("Vector Documents", rag_stats.get('total_documents', 0))
            except:
                st.metric("RAG Service", "🔴 Offline")
        
        # Configuration
        st.subheader("⚙️ Configuration")
        
        config_data = {
            "App Name": config.app_name,
            "Version": config.app_version,
            "Debug Mode": config.debug,
            "SQLite Path": config.sqlite_db_path,
            "ChromaDB Path": config.chroma_db_path,
            "Max File Size": f"{config.max_file_size_mb} MB",
            "Allowed File Types": ", ".join(config.allowed_file_types)
        }
        
        for key, value in config_data.items():
            st.text(f"{key}: {value}")
        
        # System logs (simplified)
        st.subheader("📋 Recent Activity")
        
        if 'query_history' in st.session_state:
            recent_queries = st.session_state.query_history[-5:]
            for query, response, timestamp in reversed(recent_queries):
                status = "✅" if response.get('success') else "❌"
                st.text(f"{timestamp} {status} {query[:50]}...")


def main():
    """Main application entry point."""
    app = AssetRAGInterface()
    app.run()


if __name__ == "__main__":
    main()