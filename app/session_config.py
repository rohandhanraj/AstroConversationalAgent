"""
Session Configuration Module
Allows per-session customization of agent behavior and LLM settings
"""
from typing import Dict, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    """Available LLM providers"""
    OPENAI = "openai"
    OLLAMA = "ollama"


class SessionConfig(BaseModel):
    """Configuration for a session"""
    
    # LLM Configuration
    llm_provider: LLMProvider = Field(default=LLMProvider.OLLAMA, description="LLM provider to use")
    llm_model: str = Field(default="llama3.1", description="Model name")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Temperature for generation")
    max_tokens: Optional[int] = Field(default=None, description="Max tokens in response")
    
    # Agent Configuration
    enable_hallucination_filter: bool = Field(default=True, description="Enable content filtering")
    enable_translation: bool = Field(default=True, description="Enable multi-language support")
    retrieval_k: int = Field(default=3, ge=1, le=10, description="Number of context documents to retrieve")
    
    # Response Configuration
    response_style: str = Field(default="balanced", description="Response style: concise, balanced, detailed")
    include_astrological_terms: bool = Field(default=True, description="Include Sanskrit/technical terms")
    
    # Vector Store Configuration
    vector_store: str = Field(default="faiss", description="Vector store: faiss, chroma")
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", description="Embedding model")
    chromadb_telemetry: bool = Field(default=False, description="Enable ChromaDB telemetry")
    
    # Privacy Configuration
    log_to_mongodb: bool = Field(default=True, description="Log conversations to MongoDB")
    
    class Config:
        use_enum_values = True


class SessionConfigManager:
    """
    Manages configurations for different sessions
    """
    
    def __init__(self, default_config: Optional[SessionConfig] = None):
        """
        Initialize configuration manager
        
        Args:
            default_config: Default configuration for new sessions
        """
        self.default_config = default_config or SessionConfig()
        self.session_configs: Dict[str, SessionConfig] = {}
        
        logger.info(f"SessionConfigManager initialized with default: {self.default_config.llm_provider}")
    
    def get_config(self, session_id: str) -> SessionConfig:
        """
        Get configuration for a session
        
        Args:
            session_id: Session identifier
            
        Returns:
            SessionConfig for the session
        """
        if session_id not in self.session_configs:
            # Create new config based on default
            self.session_configs[session_id] = self.default_config.copy()
            logger.debug(f"Created new config for session {session_id}")
        
        return self.session_configs[session_id]
    
    def set_config(self, session_id: str, config: SessionConfig):
        """
        Set configuration for a session
        
        Args:
            session_id: Session identifier
            config: New configuration
        """
        self.session_configs[session_id] = config
        logger.info(f"Updated config for session {session_id}: {config.dict()}")
    
    def update_config(self, session_id: str, **kwargs):
        """
        Update specific fields in session configuration
        
        Args:
            session_id: Session identifier
            **kwargs: Fields to update
        """
        config = self.get_config(session_id)
        
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                logger.debug(f"Updated {key}={value} for session {session_id}")
        
        self.session_configs[session_id] = config
    
    def delete_config(self, session_id: str):
        """Delete configuration for a session"""
        if session_id in self.session_configs:
            del self.session_configs[session_id]
            logger.debug(f"Deleted config for session {session_id}")
    
    def get_all_configs(self) -> Dict[str, SessionConfig]:
        """Get all session configurations"""
        return self.session_configs.copy()
    
    def reset_to_default(self, session_id: str):
        """Reset session to default configuration"""
        self.session_configs[session_id] = self.default_config.copy()
        logger.info(f"Reset session {session_id} to default config")


# Global session config manager
_config_manager = None


def get_config_manager() -> SessionConfigManager:
    """Get or create global configuration manager"""
    global _config_manager
    if _config_manager is None:
        _config_manager = SessionConfigManager()
    return _config_manager


# Pydantic model for API requests
class ConfigUpdateRequest(BaseModel):
    """Request model for updating session configuration"""
    llm_provider: Optional[LLMProvider] = None
    llm_model: Optional[str] = None
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = None
    enable_hallucination_filter: Optional[bool] = None
    enable_translation: Optional[bool] = None
    retrieval_k: Optional[int] = Field(None, ge=1, le=10)
    response_style: Optional[str] = None
    include_astrological_terms: Optional[bool] = None
    vector_store: Optional[str] = None
    embedding_model: Optional[str] = None
    chromadb_telemetry: Optional[bool] = None
    log_to_mongodb: Optional[bool] = None
    
    class Config:
        use_enum_values = True


# Helper functions for common configuration patterns
def create_fast_config() -> SessionConfig:
    """Create a configuration optimized for speed"""
    return SessionConfig(
        llm_provider=LLMProvider.OLLAMA,
        llm_model="llama3.1",
        temperature=0.5,
        max_tokens=500,
        enable_hallucination_filter=False,
        retrieval_k=2,
        response_style="concise"
    )


def create_detailed_config() -> SessionConfig:
    """Create a configuration for detailed responses"""
    return SessionConfig(
        llm_provider=LLMProvider.OLLAMA,
        llm_model="llama3.1",
        temperature=0.8,
        max_tokens=1000,
        enable_hallucination_filter=True,
        retrieval_k=5,
        response_style="detailed",
        include_astrological_terms=True
    )


def create_openai_config() -> SessionConfig:
    """Create a configuration using OpenAI"""
    return SessionConfig(
        llm_provider=LLMProvider.OPENAI,
        llm_model="gpt-3.5-turbo",
        temperature=0.7,
        enable_hallucination_filter=True,
        retrieval_k=3
    )


# Example usage
if __name__ == "__main__":
    # Create manager
    manager = SessionConfigManager()
    
    # Get config for new session (uses default)
    config1 = manager.get_config("session-1")
    print(f"Session 1 config: {config1.dict()}")
    
    # Update specific fields
    manager.update_config("session-1", llm_model="llama3.1", temperature=0.9)
    
    # Get updated config
    updated_config = manager.get_config("session-1")
    print(f"Updated config: {updated_config.dict()}")
    
    # Create session with preset config
    manager.set_config("session-2", create_fast_config())
    config2 = manager.get_config("session-2")
    print(f"Session 2 (fast) config: {config2.dict()}")
    
    # Show all configs
    all_configs = manager.get_all_configs()
    print(f"Total sessions: {len(all_configs)}")