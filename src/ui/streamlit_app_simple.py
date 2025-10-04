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
                    print("Parallel Agent initialized")
                    
                    st.session_state.rag_service.add_business_context()
                    print("Business context added")
                    
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
        st.caption("Natural language assistant for your data")
        
        # Two tabs: Query and Upload
        tab1, tab2 = st.tabs(["Query", "Upload Data"])
        
        with tab1:
            self.render_query_interface()
        
        with tab2:
            self.render_upload_interface()
    
    def render_query_interface(self):
        """Render the query interface."""
        st.subheader("Ask Questions About Your Files")
        
        # Example queries
        with st.expander("💡 Example Queries"):
            st.markdown("""
            **Numerical Queries:**
            - Show assets with battery voltage less than 6
            - Count assets by state
            - What's the average battery voltage?
            - Show me assets shipped after 2024-01-01
            
            **Contextual Queries:**
            - What does "In Network" state mean?
            - Explain the AT3 Pilot product
            - What actions are typically needed for assets?
            """)
        
        # Query input
        query = st.text_area(
            "Enter your question:",
            height=100,
            placeholder="e.g., Show assets with battery voltage less than 6"
        )
        
        col1, col2 = st.columns([1, 5])
        with col1:
            execute_btn = st.button("Execute Query", type="primary", use_container_width=True)
        with col2:
            if st.button("Clear", use_container_width=True):
                st.rerun()
        
        if execute_btn and query:
            with st.spinner("Processing query..."):
                try:
                    # Create query request
                    query_request = QueryRequest(query=query)
                    
                    # Execute query
                    response = asyncio.run(st.session_state.agent.query(query_request))
                    
                    if response.success:
                        # Show only the answer - hide all implementation details
                        st.success(f"Completed in {response.execution_time:.2f}s")
                        
                        # Show answer in a clean format
                        st.markdown("### Answer")
                        st.write(response.response)
                        
                        # Optionally show data table if available (without revealing it's SQL)
                        if response.data and len(response.data) > 0:
                            with st.expander(f"View Detailed Results ({len(response.data)} records)"):
                                df = pd.DataFrame(response.data)
                                st.dataframe(df, use_container_width=True, hide_index=True)
                    
                    else:
                        st.error("Unable to process your query. Please try rephrasing it.")
                
                except Exception as e:
                    st.error("An error occurred while processing your query.")
                    print(f"UI Error: {str(e)}")
    
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


def main():
    """Main entry point for the Streamlit app."""
    interface = SimpleAssetRAGInterface()
    interface.run()


if __name__ == "__main__":
    main()