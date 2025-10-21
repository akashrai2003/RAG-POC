"""
Script to load PDF documents from SMART_Logistics_KnowledgeBase_Full folder into RAG system.
"""

import sys
import os

# Add the src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from services.rag_service import RAGService


def main():
    """Load all PDF documents into the RAG system."""
    
    print("\n" + "="*80)
    print("📚 PDF RAG Loader - SMART Logistics Knowledge Base")
    print("="*80 + "\n")
    
    # Initialize RAG service
    print("🔧 Initializing RAG service...")
    rag_service = RAGService()
    
    # Get current stats
    print("\n📊 Current Collection Stats:")
    stats = rag_service.get_collection_stats()
    if 'error' not in stats:
        print(f"   • Total chunks: {stats.get('total_chunks', 0)}")
        print(f"   • Unique documents: {stats.get('unique_documents', 0)}")
        print(f"   • PDF folder: {stats.get('pdf_folder', 'N/A')}")
        
        if stats.get('source_files'):
            print(f"\n   Loaded files:")
            for file_name in stats['source_files']:
                print(f"      - {file_name}")
    
    # Ask user if they want to reload
    print("\n" + "="*80)
    choice = input("\n🔄 Do you want to (re)load all PDFs? This will clear existing data. (yes/no): ")
    
    if choice.lower() in ['yes', 'y']:
        print("\n🗑️  Clearing existing data...")
        clear_result = rag_service.clear_all_data()
        
        if not clear_result['success']:
            print(f"❌ Error clearing data: {clear_result.get('error')}")
            return
        
        print("\n📥 Loading PDF documents...")
        result = rag_service.load_pdf_documents()
        
        if result['success']:
            print(f"\n✅ SUCCESS!")
            print(f"   • Files processed: {result['files_processed']}")
            print(f"   • Total chunks created: {result['total_chunks']}")
            print(f"\n   Files loaded:")
            for file_name in result['file_names']:
                print(f"      - {file_name}")
        else:
            print(f"\n❌ ERROR: {result.get('error')}")
    else:
        print("\n✅ No changes made.")
    
    # Show final stats
    print("\n" + "="*80)
    print("📊 Final Collection Stats:")
    final_stats = rag_service.get_collection_stats()
    if 'error' not in final_stats:
        print(f"   • Total chunks: {final_stats.get('total_chunks', 0)}")
        print(f"   • Unique documents: {final_stats.get('unique_documents', 0)}")
    
    print("\n✅ PDF RAG system is ready!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
