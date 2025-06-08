from langchain_community.document_loaders import DirectoryLoader, JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from typing import List, Dict
import os
import json

class rag:
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the RAG pipeline.
        
        Args:
            data_dir: Directory containing the JSON documents to be processed
        """
        self.data_dir = data_dir
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Define persist directories for each data source
        self.persist_directories = {
            "google_docs": "faiss_index/google_docs",
            "slack_messages": "faiss_index/slack_messages",
            "teams": "faiss_index/teams",
            "company_info": "faiss_index/company_info"
        }
        
        # Create directories if they don't exist
        for directory in self.persist_directories.values():
            os.makedirs(directory, exist_ok=True)
            
        # Initialize all vector stores
        self.initialize_vector_stores()
    
    def initialize_vector_stores(self):
        """Initialize vector stores for all document types."""
        for doc_type in self.persist_directories.keys():
            # Force recreation of vector stores
            self.create_vector_store(doc_type)
    
    def load_documents_by_type(self, doc_type: str) -> List:
        """Load JSON documents of a specific type."""
        file_mapping = {
            "google_docs": "google_docs.json",
            "slack_messages": "slack_messages.json",
            "teams": "teams.json",
            "company_info": "company_info.json"
        }
        
        if doc_type not in file_mapping:
            raise ValueError(f"Unknown document type: {doc_type}")
            
        file_path = os.path.join(self.data_dir, file_mapping[doc_type])
        if not os.path.exists(file_path):
            return []
            
        loader = JSONLoader(
            file_path=file_path,
            jq_schema='.',
            text_content=False,
            json_lines=False
        )
        documents = loader.load()
        
        # Convert JSON content to string representation
        for doc in documents:
            if isinstance(doc.page_content, dict):
                doc.page_content = json.dumps(doc.page_content, indent=2)
        
        return documents
    
    def create_vector_store(self, doc_type: str):
        print(f"Creating vector store for {doc_type}")
        """Create and persist the vector store for a specific document type."""
        # Load and split documents
        documents = self.load_documents_by_type(doc_type)
        if not documents:
            print(f"No documents found for type: {doc_type}")
            return None
            
        splits = self.text_splitter.split_documents(documents)
        
        # Create and persist the FAISS vector store
        vectorstore = FAISS.from_documents(
            documents=splits,
            embedding=self.embeddings
        )
        print(f"Saving vector store for {doc_type} to {self.persist_directories[doc_type]}")
        vectorstore.save_local(self.persist_directories[doc_type])
        return vectorstore
    
    def get_retriever(self, doc_type: str, search_kwargs: dict = None):
        """Get a retriever for querying a specific vector store."""
        if doc_type not in self.persist_directories:
            raise ValueError(f"Unknown document type: {doc_type}")
            
        persist_dir = self.persist_directories[doc_type]
        
        if not os.path.exists(persist_dir):
            vectorstore = self.create_vector_store(doc_type)
            if vectorstore is None:
                raise ValueError(f"No documents found for type: {doc_type}")
        else:
            vectorstore = FAISS.load_local(
                persist_dir,
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True
            )
        
        if search_kwargs is None:
            search_kwargs = {"k": 4}
            
        return vectorstore.as_retriever(search_kwargs=search_kwargs)
    
    def get_google_docs_retriever(self, search_kwargs: dict = None):
        """Get a retriever specifically for Google Docs."""
        return self.get_retriever("google_docs", search_kwargs)
    
    def get_slack_retriever(self, search_kwargs: dict = None):
        """Get a retriever specifically for Slack messages."""
        return self.get_retriever("slack_messages", search_kwargs)
    
    def get_teams_retriever(self, search_kwargs: dict = None):
        """Get a retriever specifically for team information."""
        return self.get_retriever("teams", search_kwargs)
    
    def get_company_info_retriever(self, search_kwargs: dict = None):
        """Get a retriever specifically for company information."""
        return self.get_retriever("company_info", search_kwargs)
