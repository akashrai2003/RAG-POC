"""
Display all chunks from ChromaDB vector database.
Shows full text of each chunk along with metadata (source, page, etc.)
"""

import sys
import os

# Add the src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from services.rag_service import RAGService


def display_all_chunks(save_to_file=False, filter_source=None):
    """
    Display all chunks from ChromaDB collection.
    
    Args:
        save_to_file: If True, saves output to a text file
        filter_source: Optional - filter by source document name
    """
    
    print("\n" + "="*80)
    print("📦 ChromaDB Chunks Viewer")
    print("="*80 + "\n")
    
    # Initialize RAG service
    print("🔧 Initializing RAG service...")
    rag = RAGService()
    
    # Get collection stats
    stats = rag.get_collection_stats()
    
    if stats.get('total_chunks', 0) == 0:
        print("\n⚠️  No chunks found in database!")
        print("   Run 'python load_pdfs.py' first to load documents.")
        return
    
    print(f"✅ Found {stats['total_chunks']} chunks from {stats['unique_documents']} documents\n")
    
    # Get all data from collection
    print("📥 Retrieving all chunks from ChromaDB...")
    all_data = rag.collection.get()
    
    documents = all_data['documents']
    metadatas = all_data['metadatas']
    ids = all_data['ids']
    
    print(f"✅ Retrieved {len(documents)} chunks\n")
    
    # Filter if requested
    if filter_source:
        print(f"🔍 Filtering by source: {filter_source}")
        filtered_indices = [i for i, meta in enumerate(metadatas) if filter_source.lower() in meta.get('source', '').lower()]
        documents = [documents[i] for i in filtered_indices]
        metadatas = [metadatas[i] for i in filtered_indices]
        ids = [ids[i] for i in filtered_indices]
        print(f"✅ Filtered to {len(documents)} chunks\n")
    
    # Prepare output
    output_lines = []
    output_lines.append("="*80)
    output_lines.append("CHROMADB CHUNKS - FULL TEXT EXPORT")
    output_lines.append("="*80)
    output_lines.append(f"\nTotal Chunks: {len(documents)}")
    output_lines.append(f"Collection: {stats.get('collection_name', 'pdf_documents')}")
    output_lines.append(f"Embedding Model: {stats.get('embedding_model', 'all-MiniLM-L6-v2')}")
    output_lines.append("\n" + "="*80 + "\n")
    
    # Display each chunk
    for i, (doc_id, document, metadata) in enumerate(zip(ids, documents, metadatas), 1):
        chunk_info = []
        chunk_info.append(f"\n{'='*80}")
        chunk_info.append(f"CHUNK #{i}")
        chunk_info.append(f"{'='*80}")
        chunk_info.append(f"\nID: {doc_id}")
        chunk_info.append(f"Source: {metadata.get('source', 'Unknown')}")
        chunk_info.append(f"Page: {metadata.get('page', 'N/A')}")
        chunk_info.append(f"Chunk Index: {metadata.get('chunk', 'N/A')}")
        chunk_info.append(f"Type: {metadata.get('type', 'unknown')}")
        chunk_info.append(f"File Path: {metadata.get('file_path', 'N/A')}")
        chunk_info.append(f"Length: {len(document)} characters")
        chunk_info.append(f"\n{'-'*80}")
        chunk_info.append("FULL TEXT:")
        chunk_info.append(f"{'-'*80}\n")
        chunk_info.append(document)
        chunk_info.append(f"\n{'-'*80}")
        chunk_info.append(f"END OF CHUNK #{i}")
        chunk_info.append(f"{'-'*80}\n")
        
        chunk_text = '\n'.join(chunk_info)
        output_lines.append(chunk_text)
        
        # Print to console
        print(chunk_text)
    
    # Summary
    summary = []
    summary.append("\n" + "="*80)
    summary.append("SUMMARY")
    summary.append("="*80)
    summary.append(f"\nTotal chunks displayed: {len(documents)}")
    summary.append(f"Total characters: {sum(len(doc) for doc in documents):,}")
    summary.append(f"Average chunk size: {sum(len(doc) for doc in documents) // len(documents):,} characters")
    
    # Group by source
    sources = {}
    for meta in metadatas:
        source = meta.get('source', 'Unknown')
        sources[source] = sources.get(source, 0) + 1
    
    summary.append(f"\nChunks by source document:")
    for source, count in sorted(sources.items()):
        summary.append(f"  • {source}: {count} chunks")
    
    summary.append("\n" + "="*80 + "\n")
    
    summary_text = '\n'.join(summary)
    output_lines.append(summary_text)
    print(summary_text)
    
    # Save to file if requested
    if save_to_file:
        output_file = "chromadb_chunks_export.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_lines))
        print(f"💾 Saved to: {output_file}")
        print(f"   File size: {os.path.getsize(output_file) / 1024:.2f} KB\n")


def display_chunk_by_id(chunk_id):
    """Display a specific chunk by its ID."""
    
    print("\n" + "="*80)
    print(f"📦 Displaying Chunk: {chunk_id}")
    print("="*80 + "\n")
    
    rag = RAGService()
    
    try:
        # Get specific chunk
        result = rag.collection.get(ids=[chunk_id])
        
        if not result['documents']:
            print(f"❌ Chunk not found: {chunk_id}")
            return
        
        document = result['documents'][0]
        metadata = result['metadatas'][0]
        
        print(f"✅ Found chunk\n")
        print(f"ID: {chunk_id}")
        print(f"Source: {metadata.get('source', 'Unknown')}")
        print(f"Page: {metadata.get('page', 'N/A')}")
        print(f"Chunk Index: {metadata.get('chunk', 'N/A')}")
        print(f"Length: {len(document)} characters")
        print(f"\n{'-'*80}")
        print("FULL TEXT:")
        print(f"{'-'*80}\n")
        print(document)
        print(f"\n{'-'*80}\n")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")


def list_chunk_ids():
    """List all chunk IDs in the collection."""
    
    print("\n" + "="*80)
    print("📋 All Chunk IDs")
    print("="*80 + "\n")
    
    rag = RAGService()
    all_data = rag.collection.get()
    
    ids = all_data['ids']
    metadatas = all_data['metadatas']
    
    print(f"Total chunks: {len(ids)}\n")
    
    # Group by source
    by_source = {}
    for chunk_id, meta in zip(ids, metadatas):
        source = meta.get('source', 'Unknown')
        if source not in by_source:
            by_source[source] = []
        by_source[source].append(chunk_id)
    
    for source in sorted(by_source.keys()):
        print(f"\n📄 {source} ({len(by_source[source])} chunks):")
        for chunk_id in by_source[source]:
            print(f"   {chunk_id}")
    
    print("\n" + "="*80 + "\n")


def main():
    """Main function to handle command line arguments."""
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == '--save':
            # Save to file
            filter_source = sys.argv[2] if len(sys.argv) > 2 else None
            display_all_chunks(save_to_file=True, filter_source=filter_source)
        
        elif command == '--filter':
            # Filter by source
            if len(sys.argv) < 3:
                print("❌ Please provide a source name to filter by")
                print("   Example: python view_chunks.py --filter 'Definitions Master'")
                return
            filter_source = sys.argv[2]
            display_all_chunks(save_to_file=False, filter_source=filter_source)
        
        elif command == '--id':
            # Display specific chunk by ID
            if len(sys.argv) < 3:
                print("❌ Please provide a chunk ID")
                print("   Example: python view_chunks.py --id 'SolutionBrief_page1_chunk0'")
                return
            chunk_id = sys.argv[2]
            display_chunk_by_id(chunk_id)
        
        elif command == '--list-ids':
            # List all chunk IDs
            list_chunk_ids()
        
        elif command == '--help':
            print("\n" + "="*80)
            print("📦 ChromaDB Chunks Viewer - Help")
            print("="*80)
            print("\n🔧 Usage:")
            print("\n1. Display all chunks (console output):")
            print("   python view_chunks.py")
            print("\n2. Save all chunks to file:")
            print("   python view_chunks.py --save")
            print("\n3. Filter by source document:")
            print("   python view_chunks.py --filter 'Definitions Master'")
            print("\n4. Filter and save:")
            print("   python view_chunks.py --save 'Definitions Master'")
            print("\n5. Display specific chunk by ID:")
            print("   python view_chunks.py --id 'SolutionBrief_page1_chunk0'")
            print("\n6. List all chunk IDs:")
            print("   python view_chunks.py --list-ids")
            print("\n7. Show this help:")
            print("   python view_chunks.py --help")
            print("\n" + "="*80 + "\n")
        
        else:
            print(f"❌ Unknown command: {command}")
            print("   Run 'python view_chunks.py --help' for usage")
    
    else:
        # Default: display all chunks
        display_all_chunks(save_to_file=False)


if __name__ == "__main__":
    main()
