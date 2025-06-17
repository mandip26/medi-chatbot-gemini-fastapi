import os
import sys
import pickle
import numpy as np
from dotenv import load_dotenv

# Add app directory to path
sys.path.append('./app')

# Load environment variables
load_dotenv()

def inspect_vectorstore():
    """Inspect the existing vectorstore."""
    try:
        import faiss
        
        vectorstore_path = "app/vectorstore/db_faiss"
        index_file = os.path.join(vectorstore_path, "index.faiss")
        pkl_file = os.path.join(vectorstore_path, "index.pkl")
        
        print("=== Vectorstore Inspection ===")
        
        # Load FAISS index
        if os.path.exists(index_file):
            index = faiss.read_index(index_file)
            print(f"FAISS index dimension: {index.d}")
            print(f"Number of vectors: {index.ntotal}")
        else:
            print("FAISS index file not found")
            return
        
        # Load pickle file
        if os.path.exists(pkl_file):
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
                print(f"Pickle file keys: {list(data.keys())}")
                if 'docstore' in data:
                    print(f"Number of documents in docstore: {len(data['docstore']._dict)}")
        else:
            print("Pickle file not found")
            
    except Exception as e:
        print(f"Error inspecting vectorstore: {e}")
        import traceback
        traceback.print_exc()

def recreate_vectorstore():
    """Recreate the vectorstore with current embedding model."""
    try:
        print("\n=== Recreating Vectorstore ===")
        
        # Import required modules
        from services.chatbot_service import GoogleEmbeddings
        from langchain_community.vectorstores import FAISS
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
        
        # Load PDF documents
        print("1. Loading PDF documents...")
        data_path = "app/data"
        if not os.path.exists(data_path):
            print(f"Data path not found: {data_path}")
            return
            
        loader = DirectoryLoader(data_path, glob='*.pdf', loader_cls=PyPDFLoader)
        documents = loader.load()
        print(f"   Loaded {len(documents)} PDF pages")
        
        # Create chunks
        print("2. Creating text chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, 
            chunk_overlap=50
        )
        text_chunks = text_splitter.split_documents(documents)
        print(f"   Created {len(text_chunks)} text chunks")
        
        # Initialize current embedding model
        print("3. Initializing embedding model...")
        embeddings = GoogleEmbeddings("models/embedding-001")
        
        # Test embedding
        test_embedding = embeddings.embed_query("test")
        print(f"   Embedding dimension: {len(test_embedding)}")
        
        # Create vectorstore
        print("4. Creating vectorstore...")
        vectorstore = FAISS.from_documents(text_chunks, embeddings)
        
        # Save vectorstore
        print("5. Saving vectorstore...")
        output_path = "app/vectorstore/db_faiss"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        vectorstore.save_local(output_path)
        print(f"   Vectorstore saved to {output_path}")
        
        # Test the new vectorstore
        print("6. Testing new vectorstore...")
        test_docs = vectorstore.similarity_search("blood donation", k=2)
        print(f"   Test search returned {len(test_docs)} documents")
        if test_docs:
            print(f"   Sample: {test_docs[0].page_content[:100]}...")
        
        print("\n✓ Vectorstore recreated successfully!")
        
    except Exception as e:
        print(f"Error recreating vectorstore: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    inspect_vectorstore()
    
    # Ask if user wants to recreate
    print("\nThe vectorstore seems to have dimension mismatch.")
    print("Would you like to recreate it with the current embedding model? (y/n)")
    
    # For automation, let's recreate it
    recreate_vectorstore()
