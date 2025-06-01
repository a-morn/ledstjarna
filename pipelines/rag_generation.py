from langchain_community.document_loaders import DirectoryLoader, JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from typing import List
import os
import json

class rag:
    def __init__(self, data_dir: str = "data", persist_directory: str = "chroma_db"):
        """
        Initialize the RAG pipeline.
        
        Args:
            data_dir: Directory containing the JSON documents to be processed
            persist_directory: Directory where the vector database will be stored
        """
        self.data_dir = data_dir
        self.persist_directory = persist_directory
        # Use a more stable embedding model configuration
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
    def load_documents(self) -> List:
        """Load JSON documents from the data directory."""
        loader = DirectoryLoader(
            self.data_dir,
            glob="**/*.json",
            loader_cls=JSONLoader,
            loader_kwargs={
                'jq_schema': '.',
                'text_content': False,  # Get the raw JSON content
                'json_lines': False
            }
        )
        documents = loader.load()
        
        # Convert JSON content to string representation
        for doc in documents:
            if isinstance(doc.page_content, dict):
                doc.page_content = json.dumps(doc.page_content, indent=2)
        
        return documents
    
    def create_vector_store(self):
        """Create and persist the vector store."""
        # Load and split documents
        documents = self.load_documents()
        splits = self.text_splitter.split_documents(documents)
        
        # Create and persist the vector store
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        vectorstore.persist()
        return vectorstore
    
    def get_retriever(self, search_kwargs: dict = None):
        """Get a retriever for querying the vector store."""
        if not os.path.exists(self.persist_directory):
            vectorstore = self.create_vector_store()
        else:
            vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        
        if search_kwargs is None:
            search_kwargs = {"k": 4}
            
        return vectorstore.as_retriever(search_kwargs=search_kwargs)

def main():
    # Example usage
    rag_instance = rag()
    retriever = rag_instance.get_retriever()
    
    # Example query
    query = "What are some conflicting timelines?"
    docs = retriever.get_relevant_documents(query)
    for doc in docs:
        print(f"Content: {doc.page_content}\n")

if __name__ == "__main__":
    main() 