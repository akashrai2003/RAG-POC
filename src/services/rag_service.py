"""
PDF-based RAG service for semantic search and contextual responses.
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from typing import Dict, List, Any, Optional
import json
import os
from pathlib import Path
import logging
from config.settings import config, DatabaseConfig, QueryConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGService:
    """Service for handling PDF-based RAG queries using ChromaDB."""
    
    def __init__(self, chroma_path: str = None, pdf_folder: str = None):
        self.chroma_path = chroma_path or config.chroma_db_path
        self.pdf_folder = pdf_folder or os.path.join(os.getcwd(), "SMART_Logistics_KnowledgeBase_Full")
        self.embedding_model = SentenceTransformer(DatabaseConfig.EMBEDDING_MODEL)
        self.llm= ChatOpenAI(
            model="gpt-4o-mini",
            api_key=config.openai_api_key,
            temperature=QueryConfig.AGENT_TEMPERATURE
        )

        # Initialize ChromaDB
        self._init_chromadb()
        
        # Initialize text splitter for PDFs
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Larger chunks for PDF content
            chunk_overlap=300,  # More overlap for context preservation
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def _init_chromadb(self):
        """Initialize ChromaDB client and collection."""
        try:
            # Ensure directory exists
            os.makedirs(self.chroma_path, exist_ok=True)
            
            # Initialize ChromaDB client
            self.chroma_client = chromadb.PersistentClient(path=self.chroma_path)
            
            # Get or create collection for PDF documents
            self.collection = self.chroma_client.get_or_create_collection(
                name="pdf_documents",
                metadata={"hnsw:space": DatabaseConfig.CHROMA_DISTANCE_FUNCTION}
            )
            
            print(f"✅ ChromaDB initialized: {self.collection.count()} documents loaded")
            
        except Exception as e:
            raise Exception(f"Failed to initialize ChromaDB: {str(e)}")
    
    def load_pdf_documents(self, pdf_folder: str = None) -> Dict[str, Any]:
        """Load all PDF and DOCX documents from the specified folder into ChromaDB."""
        try:
            folder_path = pdf_folder or self.pdf_folder
            
            if not os.path.exists(folder_path):
                return {
                    'success': False,
                    'error': f"PDF folder not found: {folder_path}"
                }
            
            print(f"\n{'='*80}")
            print(f"📚 Loading documents from: {folder_path}")
            print(f"{'='*80}")
            
            documents = []
            metadatas = []
            ids = []
            
            # Find all PDF files
            pdf_files = list(Path(folder_path).glob("*.pdf"))
            all_files = pdf_files 
            
            print(f"📄 Found {len(pdf_files)} PDF files")
            
            if not all_files:
                return {
                    'success': False,
                    'error': f"No PDF files found in {folder_path}"
                }
            
            total_chunks = 0
            
            for file_path in all_files:
                try:
                    print(f"\n📖 Processing: {file_path.name}")
                    
                    # Load document based on type
                    if file_path.suffix.lower() == '.pdf':
                        loader = PyPDFLoader(str(file_path))
                    else:
                        continue
                    
                    pages = loader.load()
                    print(f"   ✓ Loaded {len(pages)} pages")
                    
                    # Process each page
                    for page_num, page in enumerate(pages):
                        # Split page content into chunks
                        chunks = self.text_splitter.split_text(page.page_content)
                        
                        for chunk_num, chunk in enumerate(chunks):
                            if len(chunk.strip()) < 50:  # Skip very small chunks
                                continue
                            
                            documents.append(chunk)
                            metadatas.append({
                                'source': file_path.name,
                                'file_path': str(file_path),
                                'page': page_num + 1,
                                'chunk': chunk_num,
                                'type': 'pdf' if file_path.suffix.lower() == '.pdf' else 'docx',
                                'total_pages': len(pages)
                            })
                            ids.append(f"{file_path.stem}_page{page_num + 1}_chunk{chunk_num}")
                            total_chunks += 1
                    
                    print(f"   ✓ Created {total_chunks - len(ids) + len(chunks)} chunks")
                    
                except Exception as e:
                    print(f"   ⚠️  Error processing {file_path.name}: {str(e)}")
                    continue
            
            if not documents:
                return {
                    'success': False,
                    'error': "No content extracted from documents"
                }
            
            # Add all documents to ChromaDB
            print(f"\n💾 Adding {len(documents)} chunks to ChromaDB...")
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            print(f"✅ Successfully loaded {len(all_files)} documents with {len(documents)} chunks")
            print(f"{'='*80}\n")
            
            return {
                'success': True,
                'files_processed': len(all_files),
                'total_chunks': len(documents),
                'file_names': [f.name for f in all_files]
            }
            
        except Exception as e:
            print(f"❌ Error loading PDF documents: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def load_single_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Load a single PDF file into the vector store."""
        try:
            if not os.path.exists(pdf_path):
                return {
                    'success': False,
                    'error': f"PDF file not found: {pdf_path}"
                }
            
            print(f"\n📖 Loading PDF: {os.path.basename(pdf_path)}")
            
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            
            documents = []
            metadatas = []
            ids = []
            
            file_name = Path(pdf_path).stem
            
            for page_num, page in enumerate(pages):
                chunks = self.text_splitter.split_text(page.page_content)
                
                for chunk_num, chunk in enumerate(chunks):
                    if len(chunk.strip()) < 50:
                        continue
                    
                    documents.append(chunk)
                    metadatas.append({
                        'source': os.path.basename(pdf_path),
                        'file_path': pdf_path,
                        'page': page_num + 1,
                        'chunk': chunk_num,
                        'type': 'pdf',
                        'total_pages': len(pages)
                    })
                    ids.append(f"{file_name}_page{page_num + 1}_chunk{chunk_num}")
            
            # Add to ChromaDB
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            print(f"✅ Loaded {len(pages)} pages, created {len(documents)} chunks")
            
            return {
                'success': True,
                'pages_loaded': len(pages),
                'chunks_created': len(documents)
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
            print(f"🤖 Generating response using with context...")
            response, token_usage = self._generate_rag_response(question, context)
            
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
                'confidence': confidence,
                'token_usage': token_usage
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
    

    
    def _build_context(self, documents: List[str], metadatas: List[Dict], distances: List[float]) -> str:
        """Build context string from search results with source citations."""
        context_parts = []
        
        for i, (doc, metadata, distance) in enumerate(zip(documents, metadatas, distances), 1):
            relevance = max(0, 1 - distance)
            source = metadata.get('source', 'Unknown')
            page = metadata.get('page', '?')
            
            # Format: [Source Number] (Source: filename, Page: X) [Relevance: 0.XX]
            # Content...
            context_parts.append(
                f"[Source {i}] (Source: {source}, Page: {page}) [Relevance: {relevance:.2f}]\n{doc}"
            )
        
        return "\n\n" + "="*80 + "\n\n".join(context_parts)
    
    def _generate_rag_response(self, question: str, context: str) -> str:
        """Generate response using LLM with retrieved context from PDFs."""
        prompt = f"""
You are an expert assistant for SMART Logistics asset tracking and supply chain management. 
Answer the user's question based on the provided context from company documentation. Ensure that if there are multiple columns or specific terminology, you use them correctly as per the context.
if there are similar terms / column names with subtle differences, pay close attention to the distinctions made in the context and have your answer reflect those distinctions accurately.
If the query has anything related to any business term then ensure that in your response mention the column names as mentioned in the context.

Context Information (from PDF documents):
{context}

User Question: {question}

Instructions:
- Take care of column names and specific terminology used in the context as there might be very subtle distinctions.
- Provide a clear, detailed answer based ONLY on the information in the context
- Cite the source documents when providing information (e.g., "According to [Source Name]...")
- If the context doesn't contain enough information, clearly state what's missing
- Focus on logistics, asset tracking, supply chain operations, and SMART platform features
- Use specific examples, metrics, and details from the context when relevant
- Be comprehensive but organized - use bullet points or sections for complex answers
- If you reference specific data or claims, indicate which source document it came from

Answer:
"""
        
        try:
            response = self.llm.invoke(prompt)
            
            # Track token usage
            input_tokens = response.response_metadata.get('token_usage', {}).get('prompt_tokens', 0)
            output_tokens = response.response_metadata.get('token_usage', {}).get('completion_tokens', 0)
            
            return response.content.strip(), {'input_tokens': input_tokens, 'output_tokens': output_tokens}
        except Exception as e:
            return f"Error generating response: {str(e)}", {'input_tokens': 0, 'output_tokens': 0}
    
    def _calculate_confidence(self, distances: List[float]) -> float:
        """Calculate confidence score based on search distances."""
        if not distances:
            return 0.0
        
        # Convert distances to relevance scores and average them
        relevance_scores = [max(0, 1 - dist) for dist in distances]
        return sum(relevance_scores) / len(relevance_scores)
    
    def _format_sources(self, documents: List[str], metadatas: List[Dict], distances: List[float]) -> List[Dict[str, Any]]:
        """Format source information for response with PDF details."""
        sources = []
        
        for doc, metadata, distance in zip(documents, metadatas, distances):
            relevance = max(0, 1 - distance)
            source = {
                'content': doc[:300] + "..." if len(doc) > 300 else doc,
                'file_name': metadata.get('source', 'Unknown'),
                'page': metadata.get('page', 'N/A'),
                'chunk': metadata.get('chunk', 0),
                'document_type': metadata.get('type', 'unknown'),
                'relevance_score': relevance,
                'file_path': metadata.get('file_path', '')
            }
            sources.append(source)
        return sources
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the PDF document collection."""
        try:
            count = self.collection.count()
            
            # Get all metadata to analyze sources
            all_data = self.collection.get()
            metadatas = all_data.get('metadatas', [])
            
            # Count unique source files
            sources = set()
            for metadata in metadatas:
                if 'source' in metadata:
                    sources.add(metadata['source'])
            
            return {
                'total_chunks': count,
                'unique_documents': len(sources),
                'collection_name': 'pdf_documents',
                'embedding_model': DatabaseConfig.EMBEDDING_MODEL,
                'pdf_folder': self.pdf_folder,
                'source_files': sorted(list(sources))
            }
            
        except Exception as e:
            return {
                'error': str(e)
            }
    
    def clear_collection(self) -> Dict[str, Any]:
        """Clear all PDF documents from the collection."""
        try:
            # Delete the collection and recreate it
            self.chroma_client.delete_collection("pdf_documents")
            self.collection = self.chroma_client.get_or_create_collection(
                name="pdf_documents",
                metadata={"hnsw:space": DatabaseConfig.CHROMA_DISTANCE_FUNCTION}
            )
            
            print("✅ PDF collection cleared successfully")
            
            return {
                'success': True,
                'message': 'PDF collection cleared successfully'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def list_loaded_documents(self) -> List[str]:
        """List all PDF documents currently loaded in the collection."""
        try:
            all_data = self.collection.get()
            metadatas = all_data.get('metadatas', [])
            
            sources = set()
            for metadata in metadatas:
                if 'source' in metadata:
                    sources.add(metadata['source'])
            
            return sorted(list(sources))
            
        except Exception as e:
            print(f"Error listing documents: {str(e)}")
            return []
    
    def search_documents(self, query: str, n_results: int = 10, filter_by_source: str = None) -> Dict[str, Any]:
        """Search for specific content in loaded PDF documents."""
        try:
            where_filter = None
            if filter_by_source:
                where_filter = {"source": filter_by_source}
            
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_filter,
                include=['documents', 'metadatas', 'distances']
            )
            
            return {
                'success': True,
                'results': results,
                'num_results': len(results['documents'][0]) if results['documents'] else 0
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def clear_all_data(self):
        """Clear all PDF data from ChromaDB collection without creating orphaned folders."""
        try:
            # Get current collection count before clearing
            old_count = self.collection.count()
            print(f"📊 Current collection has {old_count} PDF chunks")
            
            # Get all IDs and delete them (keeps same collection, no new UUID folder)
            if old_count > 0:
                # Get all document IDs (ChromaDB requires IDs for deletion)
                all_data = self.collection.get()
                all_ids = all_data['ids']
                
                # Delete all documents
                self.collection.delete(ids=all_ids)
                print(f"✅ Deleted {len(all_ids)} PDF chunks from ChromaDB collection")
            else:
                print(f"ℹ️  Collection was already empty")
            
            # Verify it's empty
            new_count = self.collection.count()
            print(f"✅ ChromaDB PDF collection cleared: pdf_documents (0 documents)")
            
            return {'success': True, 'message': f'ChromaDB cleared successfully ({old_count} PDF chunks removed)'}
        except Exception as e:
            print(f"❌ Error clearing ChromaDB: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def reload_all_pdfs(self) -> Dict[str, Any]:
        """Clear existing data and reload all PDFs from the folder."""
        try:
            print("\n🔄 Reloading all PDF documents...")
            
            # Clear existing data
            clear_result = self.clear_all_data()
            if not clear_result['success']:
                return clear_result
            
            # Load all PDFs
            load_result = self.load_pdf_documents()
            
            return load_result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }