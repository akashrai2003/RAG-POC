"""
RAG service for semantic search and contextual responses.
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import Dict, List, Any, Optional
import json
import os

from config.settings import config, DatabaseConfig, QueryConfig
from models.schemas import AssetRecord


class RAGService:
    """Service for handling RAG-based queries using ChromaDB."""
    
    def __init__(self, chroma_path: str = None):
        self.chroma_path = chroma_path or config.chroma_db_path
        self.embedding_model = SentenceTransformer(DatabaseConfig.EMBEDDING_MODEL)
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            openai_api_key=config.openai_api_key,
            temperature=QueryConfig.RAG_RESPONSE_TEMPERATURE
        )
        
        # Initialize ChromaDB
        self._init_chromadb()
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
    
    def _init_chromadb(self):
        """Initialize ChromaDB client and collection."""
        try:
            # Ensure directory exists
            os.makedirs(self.chroma_path, exist_ok=True)
            
            # Initialize ChromaDB client
            self.chroma_client = chromadb.PersistentClient(path=self.chroma_path)
            
            # Get or create collection
            self.collection = self.chroma_client.get_or_create_collection(
                name=DatabaseConfig.CHROMA_COLLECTION_NAME,
                metadata={"hnsw:space": DatabaseConfig.CHROMA_DISTANCE_FUNCTION}
            )
            
        except Exception as e:
            raise Exception(f"Failed to initialize ChromaDB: {str(e)}")
    
    def add_asset_documents(self, assets: List[AssetRecord]) -> Dict[str, Any]:
        """Add asset records as documents to the vector store."""
        try:
            documents = []
            metadatas = []
            ids = []
            
            for asset in assets:
                # Create document text from asset data
                doc_text = self._asset_to_document(asset)
                
                # Create metadata
                metadata = {
                    'asset_id': asset.asset_id,
                    'product_name': asset.product_name or '',
                    'account_name': asset.account_name or '',
                    'state_of_pallet': asset.state_of_pallet or '',
                    'current_location': asset.current_location or '',
                    'battery_voltage': asset.battery_voltage if asset.battery_voltage is not None else 0.0
                }
                
                documents.append(doc_text)
                metadatas.append(metadata)
                ids.append(asset.asset_id)
            
            # Add to ChromaDB
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            return {
                'success': True,
                'documents_added': len(documents)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def add_contextual_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Add contextual documentation to the vector store."""
        try:
            doc_texts = []
            metadatas = []
            ids = []
            
            for i, doc in enumerate(documents):
                content = doc.get('content', '')
                
                # Split long documents into chunks
                chunks = self.text_splitter.split_text(content)
                
                for j, chunk in enumerate(chunks):
                    doc_texts.append(chunk)
                    metadatas.append({
                        'source': doc.get('source', f'document_{i}'),
                        'type': doc.get('type', 'contextual'),
                        'chunk_index': j,
                        'title': doc.get('title', '')
                    })
                    ids.append(f"doc_{i}_chunk_{j}")
            
            # Add to ChromaDB
            self.collection.add(
                documents=doc_texts,
                metadatas=metadatas,
                ids=ids
            )
            
            return {
                'success': True,
                'documents_added': len(doc_texts)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def query(self, question: str, n_results: int = 5) -> Dict[str, Any]:
        """Query the RAG system for an answer."""
        try:
            print(f"\n{'='*80}")
            print(f"🔍 RAG SERVICE: Processing semantic query")
            print(f"📝 Question: {question}")
            print(f"{'='*80}")
            
            # Search for relevant documents
            print(f"🔎 Searching ChromaDB for relevant documents...")
            search_results = self.collection.query(
                query_texts=[question],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            if not search_results['documents'][0]:
                print(f"⚠️  No relevant documents found")
                print(f"{'='*80}\n")
                return {
                    'success': True,
                    'response': "I don't have enough information to answer that question.",
                    'sources': [],
                    'confidence': 0.0
                }
            
            # Extract relevant context
            context_docs = search_results['documents'][0]
            metadatas = search_results['metadatas'][0]
            distances = search_results['distances'][0]
            
            print(f"✅ Found {len(context_docs)} relevant documents")
            print(f"📊 Top distance score: {distances[0]:.4f}")
            
            # Build context string
            context = self._build_context(context_docs, metadatas, distances)
            
            # Generate response using LLM
            print(f"🤖 Generating response using OpenAI with context...")
            response = self._generate_rag_response(question, context)
            
            # Calculate confidence score
            confidence = self._calculate_confidence(distances)
            print(f"✅ Response generated (Confidence: {confidence:.2%})")
            
            # Prepare sources
            sources = self._format_sources(context_docs, metadatas, distances)
            
            print(f"{'='*80}\n")
            return {
                'success': True,
                'response': response,
                'sources': sources,
                'confidence': confidence
            }
            
        except Exception as e:
            print(f"❌ RAG SERVICE ERROR: {str(e)}")
            print(f"{'='*80}\n")
            return {
                'success': False,
                'error': str(e),
                'response': '',
                'sources': []
            }
    
    def _asset_to_document(self, asset: AssetRecord) -> str:
        """Convert asset record to searchable document text."""
        doc_parts = []
        
        # Asset ID and basic info
        doc_parts.append(f"Asset ID: {asset.asset_id}")
        
        if asset.product_name:
            doc_parts.append(f"Product: {asset.product_name}")
        
        if asset.account_name:
            doc_parts.append(f"Account: {asset.account_name}")
        
        if asset.current_location:
            doc_parts.append(f"Current Location: {asset.current_location}")
        
        if asset.state_of_pallet:
            doc_parts.append(f"State: {asset.state_of_pallet}")
        
        # Battery information
        if asset.battery_voltage is not None:
            doc_parts.append(f"Battery Voltage: {asset.battery_voltage}V")
        
        if asset.est_battery_calculate is not None:
            doc_parts.append(f"Estimated Battery: {asset.est_battery_calculate}%")
        
        # Status information
        if asset.action_needed:
            doc_parts.append(f"Action Needed: {asset.action_needed}")
        
        if asset.cardinal_tag is not None:
            doc_parts.append(f"Cardinal Tag: {'Yes' if asset.cardinal_tag else 'No'}")
        
        if asset.power_reset_occurred is not None:
            doc_parts.append(f"Power Reset Occurred: {'Yes' if asset.power_reset_occurred else 'No'}")
        
        # Date information
        if asset.date_shipped:
            doc_parts.append(f"Date Shipped: {asset.date_shipped.strftime('%Y-%m-%d')}")
        
        if asset.last_connected:
            doc_parts.append(f"Last Connected: {asset.last_connected.strftime('%Y-%m-%d %H:%M')}")
        
        if asset.power_reset_time:
            doc_parts.append(f"Power Reset Time: {asset.power_reset_time.strftime('%Y-%m-%d %H:%M')}")
        
        # Additional info
        if asset.powerup_time is not None:
            doc_parts.append(f"Powerup Time: {asset.powerup_time} seconds")
        
        if asset.account_address:
            doc_parts.append(f"Account Address: {asset.account_address}")
        
        return " | ".join(doc_parts)
    
    def _build_context(self, documents: List[str], metadatas: List[Dict], distances: List[float]) -> str:
        """Build context string from search results."""
        context_parts = []
        
        for doc, metadata, distance in zip(documents, metadatas, distances):
            # Add document with relevance score
            relevance = max(0, 1 - distance)  # Convert distance to relevance
            context_parts.append(f"[Relevance: {relevance:.2f}] {doc}")
        
        return "\n\n".join(context_parts)
    
    def _generate_rag_response(self, question: str, context: str) -> str:
        """Generate response using LLM with retrieved context."""
        prompt = f"""
        You are an expert assistant for asset tracking and management. Answer the user's question based on the provided context about asset data.
        
        Context Information:
        {context}
        
        User Question: {question}
        
        Instructions:
        - Provide a clear, informative answer based on the context
        - If the context doesn't contain enough information, say so clearly
        - Focus on asset management, tracking, and operational insights
        - Use specific examples from the context when relevant
        - Be concise but comprehensive
        
        Answer:
        """
        
        try:
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def _calculate_confidence(self, distances: List[float]) -> float:
        """Calculate confidence score based on search distances."""
        if not distances:
            return 0.0
        
        # Convert distances to relevance scores and average them
        relevance_scores = [max(0, 1 - dist) for dist in distances]
        return sum(relevance_scores) / len(relevance_scores)
    
    def _format_sources(self, documents: List[str], metadatas: List[Dict], distances: List[float]) -> List[Dict[str, Any]]:
        """Format source information for response."""
        sources = []
        
        for doc, metadata, distance in zip(documents, metadatas, distances):
            source = {
                'content': doc[:200] + "..." if len(doc) > 200 else doc,
                'metadata': metadata,
                'relevance_score': max(0, 1 - distance)
            }
            sources.append(source)
        
        return sources
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector collection."""
        try:
            count = self.collection.count()
            
            # Get sample documents
            sample_results = self.collection.peek(limit=5)
            
            return {
                'total_documents': count,
                'collection_name': DatabaseConfig.CHROMA_COLLECTION_NAME,
                'embedding_model': DatabaseConfig.EMBEDDING_MODEL,
                'sample_documents': len(sample_results.get('documents', []))
            }
            
        except Exception as e:
            return {
                'error': str(e)
            }
    
    def clear_collection(self) -> Dict[str, Any]:
        """Clear all documents from the collection."""
        try:
            # Delete the collection and recreate it
            self.chroma_client.delete_collection(DatabaseConfig.CHROMA_COLLECTION_NAME)
            self.collection = self.chroma_client.get_or_create_collection(
                name=DatabaseConfig.CHROMA_COLLECTION_NAME,
                metadata={"hnsw:space": DatabaseConfig.CHROMA_DISTANCE_FUNCTION}
            )
            
            return {
                'success': True,
                'message': 'Collection cleared successfully'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def add_business_context(self):
        """Add business context documents about asset management."""
        business_docs = [
            {
                'title': 'Asset State Definitions',
                'content': '''
                Asset State Definitions:
                
                In Network: Asset is actively connected and operating within the tracking network. 
                These assets are providing regular status updates and location information.
                
                In Transit: Asset is currently being transported between locations. May have 
                limited connectivity during transportation but should reconnect upon arrival.
                
                Returned for Refurbishment - DEACTIVATED: Asset has been returned to the facility 
                for maintenance, repair, or refurbishment. Asset is temporarily deactivated.
                
                Deactivated Tags - Returned/retired: Assets that have been permanently deactivated 
                and returned to inventory or retired from service.
                
                Deactivated Tags - Not Returned: Assets that have been deactivated but have not 
                yet been physically returned to the facility.
                ''',
                'source': 'business_documentation',
                'type': 'definitions'
            },
            {
                'title': 'Product Information',
                'content': '''
                Product Types:
                
                AT3 Pilot: Third-generation asset tracking device with pilot program features. 
                Designed for comprehensive asset monitoring with enhanced battery life and 
                connectivity options.
                
                AT5 - Bracket: Fifth-generation asset tracking device with bracket mounting system. 
                Provides improved durability and installation flexibility for various asset types.
                
                Battery Voltage: Critical monitoring parameter. Normal operating range is typically 
                6V to 12V. Voltages below 6V may indicate need for battery replacement or charging.
                
                Power Reset: Indicates whether the device has experienced a power cycle or reset. 
                Can be caused by maintenance, battery issues, or system updates.
                ''',
                'source': 'product_documentation',
                'type': 'product_info'
            },
            {
                'title': 'Asset Management Processes',
                'content': '''
                Asset Management Processes:
                
                Shipping Process: Assets are assigned to accounts and shipped to customer locations. 
                Date_Shipped tracks when the asset left the facility.
                
                Connection Monitoring: Last_Connected timestamp shows when the asset last 
                communicated with the tracking system. Extended periods without connection 
                may indicate connectivity issues or device problems.
                
                Action Needed: Field indicating required actions such as "RMA to SMART" 
                (Return Merchandise Authorization to SMART facility) for maintenance or replacement.
                
                Cardinal Tag: Indicates whether the asset is part of the Cardinal tracking system, 
                which provides enhanced monitoring and management capabilities.
                
                Location Tracking: Current_Location_Name shows the asset's last known location, 
                which may be a customer site, distribution center, or in-transit status.
                ''',
                'source': 'process_documentation',
                'type': 'processes'
            }
        ]
        
        return self.add_contextual_documents(business_docs)
    
    def clear_all_data(self):
        """Clear all data from ChromaDB collection without creating orphaned folders."""
        try:
            # Get current collection count before clearing
            old_count = self.collection.count()
            print(f"📊 Current collection has {old_count} documents")
            
            # Get all IDs and delete them (keeps same collection, no new UUID folder)
            if old_count > 0:
                # Peek at all document IDs (ChromaDB requires IDs for deletion)
                all_data = self.collection.get()
                all_ids = all_data['ids']
                
                # Delete all documents
                self.collection.delete(ids=all_ids)
                print(f"✅ Deleted {len(all_ids)} documents from ChromaDB collection")
            else:
                print(f"ℹ️  Collection was already empty")
            
            # Verify it's empty
            new_count = self.collection.count()
            print(f"✅ ChromaDB collection cleared: {DatabaseConfig.CHROMA_COLLECTION_NAME} (0 documents)")
            
            return {'success': True, 'message': f'ChromaDB cleared successfully ({old_count} documents removed)'}
        except Exception as e:
            print(f"❌ Error clearing ChromaDB: {str(e)}")
            return {'success': False, 'error': str(e)}