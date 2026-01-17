"""
Retrieval Module
Handles document loading, embedding, and retrieval using FAISS or ChromaDB
"""
import os
import json
from typing import List, Dict, Optional, Tuple
from pathlib import Path

# Disable ChromaDB telemetry if specified
if os.getenv('ANONYMIZED_TELEMETRY', 'False').lower() == 'false':
    os.environ['ANONYMIZED_TELEMETRY'] = 'False'

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

# Vector store imports
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma


class RetrievalEngine:
    """
    Retrieval engine for astrological knowledge base
    Supports both FAISS and ChromaDB
    """
    
    def __init__(
        self,
        data_dir: str = "./data",
        vector_store_type: str = "faiss",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        persist_directory: Optional[str] = None,
        use_openai_embeddings: bool = False
    ):
        """
        Initialize retrieval engine
        
        Args:
            data_dir: Directory containing knowledge base files
            vector_store_type: 'faiss' or 'chroma'
            embedding_model: Name of embedding model
            persist_directory: Directory to persist vector store
            use_openai_embeddings: Whether to use OpenAI embeddings
        """
        self.data_dir = Path(data_dir)
        self.vector_store_type = vector_store_type.lower()
        self.persist_directory = persist_directory
        
        # Initialize embeddings
        if use_openai_embeddings:
            self.embeddings = OpenAIEmbeddings()
        else:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model,
                model_kwargs={'device': 'cpu'}
            )
        
        # Initialize vector store
        self.vector_store = None
        self.documents = []
        
    def load_documents(self) -> List[Document]:
        """
        Load documents from data directory
        
        Returns:
            List of Document objects
        """
        documents = []
        
        # Load text files
        txt_files = list(self.data_dir.glob("*.txt"))
        for txt_file in txt_files:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read()
                doc = Document(
                    page_content=content,
                    metadata={"source": txt_file.name, "type": "text"}
                )
                documents.append(doc)
        
        # Load JSON files
        json_files = list(self.data_dir.glob("*.json"))
        for json_file in json_files:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Convert JSON to text documents
                if "planetary_traits" in data:
                    for planet, traits in data["planetary_traits"].items():
                        content = f"Planet: {planet}\n"
                        content += f"Keywords: {', '.join(traits.get('keywords', []))}\n"
                        content += f"Positive traits: {', '.join(traits.get('positive', []))}\n"
                        content += f"Negative traits: {', '.join(traits.get('negative', []))}\n"
                        content += f"Career influences: {', '.join(traits.get('career_influences', []))}\n"
                        
                        doc = Document(
                            page_content=content,
                            metadata={"source": json_file.name, "type": "planetary_trait", "planet": planet}
                        )
                        documents.append(doc)
                
                if "zodiac_signs" in data:
                    for sign, info in data["zodiac_signs"].items():
                        content = f"Zodiac Sign: {sign}\n"
                        content += f"Element: {info.get('element', '')}\n"
                        content += f"Ruling Planet: {info.get('ruling_planet', '')}\n"
                        content += f"Personality: {', '.join(info.get('personality_traits', []))}\n"
                        content += f"Career Strength: {info.get('career_strength', '')}\n"
                        content += f"Love Style: {info.get('love_style', '')}\n"
                        
                        doc = Document(
                            page_content=content,
                            metadata={"source": json_file.name, "type": "zodiac", "sign": sign}
                        )
                        documents.append(doc)
        
        self.documents = documents
        return documents
    
    def split_documents(self, documents: List[Document], chunk_size: int = 500, chunk_overlap: int = 50) -> List[Document]:
        """
        Split documents into chunks
        
        Args:
            documents: List of documents to split
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of split documents
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        
        splits = text_splitter.split_documents(documents)
        return splits
    
    def build_vector_store(self, documents: Optional[List[Document]] = None):
        """
        Build vector store from documents
        
        Args:
            documents: Optional list of documents (will load if not provided)
        """
        if documents is None:
            documents = self.load_documents()
        
        # Split documents
        all_splits = self.split_documents(documents)
        
        print(f"Building vector store with {len(all_splits)} document chunks...")
        
        if self.vector_store_type == "faiss":
            # Build FAISS vector store
            embedding_dim = len(self.embeddings.embed_query("hello world"))
            index = faiss.IndexFlatL2(embedding_dim)
            
            self.vector_store = FAISS(
                embedding_function=self.embeddings,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
            )
            
            ids = self.vector_store.add_documents(documents=all_splits)
            print(f"Added {len(ids)} documents to FAISS vector store")
            
            # Save if persist directory specified
            if self.persist_directory:
                os.makedirs(self.persist_directory, exist_ok=True)
                self.vector_store.save_local(self.persist_directory)
                print(f"FAISS index saved to {self.persist_directory}")
        
        elif self.vector_store_type == "chroma":
            # Build ChromaDB vector store
            persist_dir = self.persist_directory or "./chroma_db"
            
            self.vector_store = Chroma(
                collection_name="astro_knowledge",
                embedding_function=self.embeddings,
                persist_directory=persist_dir,
            )
            
            ids = self.vector_store.add_documents(documents=all_splits)
            print(f"Added {len(ids)} documents to ChromaDB")
        
        else:
            raise ValueError(f"Unsupported vector store type: {self.vector_store_type}")
    
    def load_vector_store(self):
        """Load existing vector store from disk"""
        if not self.persist_directory or not os.path.exists(self.persist_directory):
            raise ValueError("Persist directory not found. Build vector store first.")
        
        if self.vector_store_type == "faiss":
            self.vector_store = FAISS.load_local(
                self.persist_directory,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"FAISS vector store loaded from {self.persist_directory}")
        
        elif self.vector_store_type == "chroma":
            self.vector_store = Chroma(
                collection_name="astro_knowledge",
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
            )
            print(f"ChromaDB loaded from {self.persist_directory}")
    
    def retrieve(self, query: str, k: int = 3, filter_dict: Optional[Dict] = None) -> List[Tuple[Document, float]]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: Query string
            k: Number of documents to retrieve
            filter_dict: Optional metadata filters
            
        Returns:
            List of (Document, score) tuples
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Build or load it first.")
        
        # Retrieve with similarity scores
        results = self.vector_store.similarity_search_with_score(query, k=k)
        
        # Filter by metadata if provided
        if filter_dict:
            filtered_results = []
            for doc, score in results:
                if all(doc.metadata.get(key) == value for key, value in filter_dict.items()):
                    filtered_results.append((doc, score))
            return filtered_results
        
        return results
    
    def retrieve_context(self, query: str, user_profile: Dict, k: int = 3) -> List[str]:
        """
        Retrieve context relevant to query and user profile
        
        Args:
            query: User query
            user_profile: User's astrological profile
            k: Number of documents to retrieve
            
        Returns:
            List of context strings
        """
        # Enhance query with user profile
        enhanced_query = f"{query} {user_profile.get('sun_sign', '')}"
        
        results = self.retrieve(enhanced_query, k=k)
        
        context_list = []
        for doc, score in results:
            context_list.append(doc.page_content)
        
        return context_list
    
    def get_retriever(self, k: int = 3):
        """Get LangChain retriever interface"""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        
        return self.vector_store.as_retriever(search_kwargs={"k": k})


# Example usage and testing
if __name__ == "__main__":
    # Initialize retrieval engine
    engine = RetrievalEngine(
        data_dir="../data",
        vector_store_type="faiss",
        persist_directory="./vector_store"
    )
    
    # Build vector store
    engine.build_vector_store()
    
    # Test retrieval
    query = "What are Leo's career strengths?"
    results = engine.retrieve(query, k=3)
    
    print(f"\nQuery: {query}")
    print("\nTop Results:")
    for i, (doc, score) in enumerate(results, 1):
        print(f"\n{i}. Score: {score:.4f}")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        print(f"Content: {doc.page_content[:200]}...")
