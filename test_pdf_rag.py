"""
Test script for PDF-based RAG system.
Query the knowledge base and see results with sources.
"""

import sys
import os

# Add the src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from services.rag_service import RAGService


def format_sources(sources):
    """Format source information for display."""
    print("\n📚 Sources:")
    print("-" * 80)
    
    for i, source in enumerate(sources, 1):
        print(f"\n[{i}] {source['file_name']} (Page {source['page']})")
        print(f"    Relevance Score: {source['relevance_score']:.2%}")
        print(f"    Content Preview: {source['content'][:150]}...")


def main():
    """Run interactive PDF RAG query session."""
    
    print("\n" + "="*80)
    print("🤖 PDF RAG Query System - SMART Logistics Knowledge Base")
    print("="*80 + "\n")
    
    # Initialize RAG service
    print("🔧 Initializing RAG service...")
    rag_service = RAGService()
    
    # Check if documents are loaded
    stats = rag_service.get_collection_stats()
    if stats.get('total_chunks', 0) == 0:
        print("\n⚠️  WARNING: No documents loaded!")
        print("   Please run 'python load_pdfs.py' first to load PDF documents.")
        return
    
    print(f"✅ Loaded: {stats['unique_documents']} documents with {stats['total_chunks']} chunks")
    
    # Sample questions
    sample_questions = [
        "What is SMART Logistics?",
        "What are the key features of the asset tracking system?",
        "What are the different asset states and their meanings?",
        "How does the battery monitoring work?",
        "What is FRIG and how does it work?",
        "Explain the reporting definitions and metrics"
    ]
    
    print("\n" + "="*80)
    print("💡 Sample Questions:")
    for i, q in enumerate(sample_questions, 1):
        print(f"   {i}. {q}")
    
    print("\n" + "="*80)
    print("\nEnter your question (or 'quit' to exit, 'list' to see documents):")
    
    while True:
        try:
            print("\n" + "-"*80)
            question = input("\n❓ Your question: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if question.lower() == 'list':
                print("\n📄 Loaded Documents:")
                for doc in stats.get('source_files', []):
                    print(f"   • {doc}")
                continue
            
            # Query the RAG system
            print("\n🔍 Searching knowledge base...")
            result = rag_service.query(question, n_results=5)
            
            if result['success']:
                print("\n" + "="*80)
                print("💡 ANSWER:")
                print("="*80)
                print(f"\n{result['response']}")
                print(f"\n📊 Confidence: {result['confidence']:.2%}")
                
                if result['sources']:
                    format_sources(result['sources'])
            else:
                print(f"\n❌ Error: {result.get('error')}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")


if __name__ == "__main__":
    main()
