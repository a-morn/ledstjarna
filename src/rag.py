from langchain_community.document_loaders import DirectoryLoader, JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from typing import List
import os
import json

class rag:
    def __init__(self, data_dir: str = "data", persist_directory: str = "faiss_index"):
        """
        Initialize the RAG pipeline.
        
        Args:
            data_dir: Directory containing the JSON documents to be processed
            persist_directory: Directory where the vector database will be stored
        """
        self.data_dir = data_dir
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
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
        
        # Create and persist the FAISS vector store
        vectorstore = FAISS.from_documents(
            documents=splits,
            embedding=self.embeddings
        )
        vectorstore.save_local(self.persist_directory)
        return vectorstore
    
    def get_retriever(self, search_kwargs: dict = None):
        """Get a retriever for querying the vector store."""
        if not os.path.exists(self.persist_directory):
            vectorstore = self.create_vector_store()
        else:
            vectorstore = FAISS.load_local(
                self.persist_directory,
                embeddings=self.embeddings
            )
        
        if search_kwargs is None:
            search_kwargs = {"k": 4}
            
        return vectorstore.as_retriever(search_kwargs=search_kwargs)
