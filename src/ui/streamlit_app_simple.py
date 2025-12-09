"""
Simplified Streamlit web interface for the Asset RAG POC.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import json
import asyncio
from typing import Optional

# Import our services
from config.settings import config
from services.sql_service import SQLService
from services.rag_service import RAGService
from services.data_service import DataProcessingService
from services.agent_service_parallel import ParallelAssetRAGAgent
from models.schemas import QueryRequest


class SimpleAssetRAGInterface:
    """Simplified Streamlit interface for the Asset RAG system."""
    
    def __init__(self):
        self.setup_page_config()
        self.initialize_services()
    
    def setup_page_config(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="Query System",
            page_icon="🏷️",
            layout="wide",
            initial_sidebar_state="collapsed"
        )
        
        # Minimal CSS
        st.markdown("""
        <style>
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 8px 16px;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def initialize_services(self):
        """Initialize all services."""
        if 'services_initialized' not in st.session_state:
            with st.spinner("Initializing services..."):
                try:
                    print("\n" + "="*80)
                    print("INITIALIZING SERVICES")
                    print("="*80)
                    
                    st.session_state.sql_service = SQLService()
                    print("SQL Service initialized")
                    
                    st.session_state.rag_service = RAGService()
                    print("RAG Service initialized")
                    
                    st.session_state.data_service = DataProcessingService(
                        st.session_state.sql_service,
                        st.session_state.rag_service
                    )
                    print("Data Service initialized")
                    
                    st.session_state.agent = ParallelAssetRAGAgent(
                        st.session_state.sql_service,
                        st.session_state.rag_service
                    )
                    print("Agent initialized (fast execution)")
                    
                    # Note: Business context is now loaded from PDFs, not hard-coded
                    # st.session_state.rag_service.add_business_context()
                    # print("Business context added")
                    
                    st.session_state.services_initialized = True
                    print("="*80 + "\n")
                    
                    st.success("Services initialized successfully!")
                    
                except Exception as e:
                    print(f"Service initialization failed: {str(e)}")
                    print("="*80 + "\n")
                    st.error(f"Failed to initialize services: {str(e)}")
                    st.stop()
    
    def run(self):
        """Main application runner."""
        # Simple header
        st.title("RAG PoC")
        st.caption("Natural language assistant for Einstein Data Cloud")
        
        # Info banner about Einstein Analytics and optimization
        col1, col2 = st.columns([2, 1])
        with col1:
            st.info("🔗 Connected to Salesforce Einstein Data Cloud - querying live data directly!")
        with col2:
            st.success("⚡ Parallel Execution: Fast query processing")
        
        # Tabs for Query and Upload
        tab1, tab2 = st.tabs(["🔍 Query", "📚 Upload PDFs"])
        
        with tab1:
            self.render_query_interface()
        
        with tab2:
            self.render_pdf_upload_interface()
    
    def render_query_interface(self):
        """Render the query interface."""
        st.subheader("Ask Questions About Your Data & Documentation")
        
        # Example queries with smart routing indicators
        with st.expander("💡 Example Queries"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📊 Data Queries**")
                st.markdown("""
                - Show assets with battery voltage less than 6
                - Count assets by state
                - List all assets with Cardinal Tags
                - Show me assets shipped after 2024-01-01
                - What's the average battery voltage?
                """)
            
            with col2:
                st.markdown("**📚 Documentation Queries**")
                st.markdown("""
                - What is a Cardinal Tag?
                - Explain the "In Transit" state
                - What does FRIG mean?
                - Describe the AT3 Pilot product
                - What actions are typically needed?
                """)
        
        # Query input
        query = st.text_area(
            "Enter your question:",
            height=100,
            placeholder="e.g., Show assets with battery voltage less than 6"
        )
        
        # Optional account filter
        with st.expander("🔐 Advanced: Account Filtering (Optional)"):
            account_id = st.text_input(
                "Salesforce Account ID",
                placeholder="e.g., 001xx000003DGbQAAW",
                help="Filter results to a specific Salesforce account. Leave empty to query all accounts."
            )
            if account_id:
                st.info(f"Results will be filtered to Account ID: {account_id}")
        
        col1, col2 = st.columns([1, 5])
        with col1:
            execute_btn = st.button("Execute Query", type="primary", use_container_width=True)
        with col2:
            if st.button("Clear", use_container_width=True):
                st.rerun()
        
        if execute_btn and query:
            # Get account_id if it was entered, otherwise None
            filter_account_id = account_id.strip() if 'account_id' in locals() and account_id else None
            
            with st.spinner("Processing query..."):
                try:
                    # Create query request with optional account filter
                    query_request = QueryRequest(
                        query=query,
                        account_id=filter_account_id
                    )
                    
                    # Execute query
                    response = asyncio.run(st.session_state.agent.query(query_request))
                    
                    if response.success:
                        # Apply post-processing filter to data
                        filtered_data = _filter_response_data(response.data) if response.data else None
                        
                        # Show completion metrics
                        query_type = response.query_type
                        
                        # Create status message based on query type
                        if query_type == "rag":
                            status_icon = "📚"
                            status_text = "Documentation query"
                        elif query_type == "sql":
                            status_icon = "📊"
                            status_text = "Data query"
                        else:
                            status_icon = "🔄"
                            status_text = "Hybrid query"
                        
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.success(f"{status_icon} {status_text}")
                        with col2:
                            st.metric("Time", f"{response.execution_time:.2f}s")
                        
                        # Show account filter indicator
                        if filter_account_id:
                            st.caption(f"🔐 Filtered by Account: {filter_account_id}")
                        
                        # Show answer in a clean format
                        st.markdown("### Answer")
                        st.write(response.response)
                        
                        # Optionally show data table if available
                        if filtered_data and len(filtered_data) > 0:
                            with st.expander(f"📋 View Detailed Results ({len(filtered_data)} records)"):
                                df = pd.DataFrame(filtered_data)
                                st.dataframe(df, use_container_width=True, hide_index=True)
                    
                    else:
                        st.error("Unable to process your query. Please try rephrasing it.")
                
                except Exception as e:
                    st.error("An error occurred while processing your query.")
                    print(f"UI Error: {str(e)}")
    
    def render_pdf_upload_interface(self):
        """Render the PDF upload interface for RAG knowledge base."""
        st.subheader("Upload PDF Documents to Knowledge Base")
        
        # Show current stats
        with st.expander("📊 Current Knowledge Base Stats"):
            try:
                stats = st.session_state.rag_service.get_collection_stats()
                if 'error' not in stats:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Chunks", stats.get('total_chunks', 0))
                    with col2:
                        st.metric("Unique Documents", stats.get('unique_documents', 0))
                    
                    if stats.get('source_files'):
                        st.write("**Loaded Documents:**")
                        for file_name in stats['source_files']:
                            st.text(f"• {file_name}")
                else:
                    st.error(f"Error loading stats: {stats['error']}")
            except Exception as e:
                st.error(f"Error loading stats: {str(e)}")
        
        st.markdown("---")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload PDF documents to add to the knowledge base",
            key="pdf_uploader"
        )
        
        if uploaded_files:
            st.info(f"📄 Selected {len(uploaded_files)} file(s)")
            
            # Show selected files
            with st.expander("📋 Selected Files"):
                for file in uploaded_files:
                    st.text(f"• {file.name} ({file.size / 1024:.1f} KB)")
            
            st.markdown("---")
            
            # Upload mode selection
            clear_db = st.checkbox(
                "🗑️ Clear existing knowledge base before uploading",
                value=False,
                help="If checked, all existing PDF chunks will be removed before adding new files. If unchecked, new files will be added/updated."
            )
            
            if clear_db:
                st.warning("⚠️ This will remove all existing PDF chunks from the knowledge base!")
            else:
                st.info("ℹ️ Files with the same name will be updated. New files will be added.")
            
            col1, col2 = st.columns([1, 4])
            with col1:
                process_btn = st.button("Upload PDFs", type="primary", use_container_width=True)
            
            if process_btn:
                with st.spinner("Processing PDF files..."):
                    try:
                        print("\n" + "="*80)
                        print(f"UPLOADING {len(uploaded_files)} PDF FILES")
                        print(f"Clear DB: {clear_db}")
                        print("="*80)
                        
                        # Prepare files for processing
                        files_data = []
                        for uploaded_file in uploaded_files:
                            content = uploaded_file.read()
                            files_data.append((content, uploaded_file.name))
                            uploaded_file.seek(0)  # Reset file pointer
                        
                        # Process files using RAG service
                        result = st.session_state.rag_service.load_multiple_pdfs_from_bytes(
                            files_data,
                            clear_db=clear_db
                        )
                        
                        if result['success']:
                            st.success(f"✅ Successfully processed {result['files_processed']}/{result['total_files']} files")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Files Processed", f"{result['files_processed']}/{result['total_files']}")
                            with col2:
                                st.metric("Chunks Created", result['total_chunks'])
                            
                            # Show failed files if any
                            if result.get('failed_files'):
                                with st.expander("⚠️ Failed Files"):
                                    for failed in result['failed_files']:
                                        st.error(f"**{failed['filename']}**: {failed['error']}")
                            
                            # Show updated stats
                            st.markdown("---")
                            st.markdown("**Updated Knowledge Base:**")
                            stats = st.session_state.rag_service.get_collection_stats()
                            if 'error' not in stats:
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Total Chunks", stats.get('total_chunks', 0))
                                with col2:
                                    st.metric("Unique Documents", stats.get('unique_documents', 0))
                        else:
                            st.error(f"Upload failed: {result.get('error', 'Unknown error')}")
                    
                    except Exception as e:
                        st.error(f"Error uploading PDFs: {str(e)}")
                        print(f"PDF Upload Error: {str(e)}")
    
    def render_upload_interface(self):
        """Render the data upload interface."""
        st.subheader("Upload Asset Data")
        
        uploaded_file = st.file_uploader(
            "Choose an Excel or CSV file",
            type=['xlsx', 'xls', 'csv'],
            help="Upload asset data to populate the database",
            key="file_uploader"
        )
        
        if uploaded_file is not None:
            # Upload mode selection
            st.markdown("---")
            upload_mode = st.radio(
                "Upload Mode:",
                options=["Append to existing data", "Replace all existing data"],
                index=0,
                help="Choose whether to add to existing data or replace it completely",
                key="upload_mode"
            )
            
            # Show warning for replace mode
            if upload_mode == "Replace all existing data":
                st.warning("⚠️ This will delete all existing asset data from the database!")
            
            if st.button("Process File", type="primary", key="process_btn"):
                with st.spinner("Processing file..."):
                    try:
                        print("\n" + "="*80)
                        print(f"UPLOADING FILE: {uploaded_file.name}")
                        print(f"Mode: {upload_mode}")
                        print("="*80)
                        
                        # Clear databases if replace mode is selected
                        if upload_mode == "Replace all existing data":
                            with st.spinner("Clearing existing data..."):
                                print("🗑️  Clearing existing databases...")
                                self._clear_databases()
                                st.info("✅ Existing data cleared")
                        
                        result = st.session_state.data_service.process_uploaded_file(
                            uploaded_file,
                            uploaded_file.name
                        )
                        
                        if result.success:
                            mode_text = "added" if upload_mode == "Append to existing data" else "loaded"
                            st.success(f"Successfully {mode_text} {result.records_processed} records in {result.processing_time:.2f}s")
                            
                            # Use 'errors' instead of 'validation_errors'
                            if result.errors:
                                with st.expander("⚠️ Issues Encountered"):
                                    for error in result.errors:
                                        st.warning(error)
                        else:
                            st.error(f"Upload failed: {result.errors[0] if result.errors else 'Unknown error'}")
                    
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        print(f"Upload Error: {str(e)}")
    
    def _clear_databases(self):
        """Clear both SQLite and ChromaDB databases."""
        try:
            # Clear SQLite
            st.session_state.sql_service.clear_all_data()
            print("✅ SQLite database cleared")
            
            # Clear ChromaDB
            st.session_state.rag_service.clear_all_data()
            print("✅ ChromaDB cleared")
            
        except Exception as e:
            print(f"❌ Error clearing databases: {str(e)}")
            raise


def _filter_response_data(data):
    """
    Filter out redundant count-only data that should not be shown in UI tables.
    
    Logic:
    - If all rows have identical values AND only count-type columns exist -> filter out
    - If it's grouped data (count per category) -> keep it (useful aggregation)
    - If mixed data types -> keep it
    
    Returns None if data should be filtered out, otherwise returns the data.
    """
    if not data or len(data) == 0:
        return data
    
    # Check if this looks like redundant count data
    if _is_redundant_count_data(data):
        print(f"🚫 Filtering out redundant count data ({len(data)} identical rows)")
        return None
    
    return data


def _is_redundant_count_data(data):
    """
    Detect if data consists of redundant count rows that add no value.
    Optimized for large datasets - uses sampling and early exit.
    
    Criteria for filtering:
    1. All rows are identical AND
    2. Only contains count-type columns AND  
    3. All count values are 1 (indicating row-level counts, not aggregations)
    """
    if len(data) < 2:  # Single row or empty - don't filter
        return False
    
    first_row = data[0]
    
    # Early exit: Check if only count-type columns exist (fast check)
    count_columns = [col for col in first_row.keys() 
                    if any(keyword in col.lower() 
                          for keyword in ['count', 'total', 'sum', 'avg', 'average', 'min', 'max'])]
    
    if len(count_columns) != len(first_row):
        return False  # Has non-count columns - keep the data
    
    # Early exit: Check if all count values are 1 in first row
    for col in count_columns:
        value = first_row[col]
        if not (isinstance(value, (int, float)) and value == 1):
            return False  # Not a redundant count of 1 - keep the data
    
    # For large datasets, use sampling instead of checking every row
    if len(data) > 1000:
        # Sample check: Check first 100, middle 50, and last 50 rows
        sample_indices = (
            list(range(min(100, len(data)))) +  # First 100
            list(range(len(data)//2 - 25, len(data)//2 + 25)) +  # Middle 50
            list(range(max(0, len(data) - 50), len(data)))  # Last 50
        )
        # Remove duplicates and limit
        sample_indices = list(set(sample_indices))[:200]
        
        # Check sampled rows for identical structure
        for i in sample_indices:
            if data[i] != first_row:
                return False  # Sample shows variation - keep the data
        
        print(f"🔍 Large dataset detected ({len(data)} rows) - used sampling for redundancy check")
    else:
        # Small dataset - check all rows (but with early exit)
        for i in range(1, min(len(data), 1000)):  # Cap at 1000 for safety
            if data[i] != first_row:
                return False  # Rows differ - keep the data
    
    # All criteria met - this is redundant count data
    return True


def main():
    """Main entry point for the Streamlit app."""
    interface = SimpleAssetRAGInterface()
    interface.run()


if __name__ == "__main__":
    main()