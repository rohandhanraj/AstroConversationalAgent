"""
LLM Wrapper Module
Provides abstraction over different LLM providers (OpenAI, Ollama)
"""
from typing import Optional, Dict, Any
from enum import Enum
import os
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage


class LLMProvider(Enum):
    """Enum for LLM providers"""
    OPENAI = "openai"
    OLLAMA = "ollama"


class LLMWrapper:
    """
    Wrapper class for LLM operations with support for multiple providers
    """
    
    def __init__(
        self,
        provider: str = "openai",
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        """
        Initialize LLM wrapper
        
        Args:
            provider: LLM provider ('openai' or 'ollama')
            model_name: Name of the model to use
            temperature: Temperature for generation
            max_tokens: Maximum tokens in response
            api_key: API key for OpenAI (if applicable)
            base_url: Base URL for Ollama (if applicable)
        """
        self.provider = LLMProvider(provider.lower())
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize the appropriate LLM
        if self.provider == LLMProvider.OPENAI:
            self.model_name = model_name or "gpt-3.5-turbo"
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            
            self.llm = ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                api_key=self.api_key
            )
            
        elif self.provider == LLMProvider.OLLAMA:
            self.model_name = model_name or "llama3.1"
            self.base_url = base_url or "http://localhost:11434"
            
            self.llm = ChatOllama(
                model=self.model_name,
                temperature=self.temperature,
                base_url=self.base_url
            )
        
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def invoke(self, messages: list) -> str:
        """
        Invoke the LLM with messages
        
        Args:
            messages: List of message dictionaries or BaseMessage objects
            
        Returns:
            Generated response text
        """
        try:
            # Convert dict messages to BaseMessage if needed
            formatted_messages = []
            for msg in messages:
                if isinstance(msg, dict):
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    
                    if role == "system":
                        formatted_messages.append(SystemMessage(content=content))
                    elif role == "assistant":
                        formatted_messages.append(AIMessage(content=content))
                    else:
                        formatted_messages.append(HumanMessage(content=content))
                else:
                    formatted_messages.append(msg)
            
            response = self.llm.invoke(formatted_messages)
            return response.content
            
        except Exception as e:
            print(f"Error invoking LLM: {e}")
            return f"Error: {str(e)}"
    
    async def ainvoke(self, messages: list) -> str:
        """
        Async invoke the LLM with messages
        
        Args:
            messages: List of message dictionaries or BaseMessage objects
            
        Returns:
            Generated response text
        """
        try:
            # Convert dict messages to BaseMessage if needed
            formatted_messages = []
            
            response = await self.llm.ainvoke(formatted_messages)
            return response.content
            
        except Exception as e:
            print(f"Error in async invoke: {e}")
            return f"Error: {str(e)}"
    
    async def astream(self, messages: list):
        """
        Stream responses from the LLM
        
        Args:
            messages: List of message dictionaries or BaseMessage objects
            
        Yields:
            Response chunks
        """
        try:
            # Convert dict messages to BaseMessage if needed
            formatted_messages = []
            for msg in messages:
                if isinstance(msg, dict):
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    
                    if role == "system":
                        formatted_messages.append(SystemMessage(content=content))
                    elif role == "assistant":
                        formatted_messages.append(AIMessage(content=content))
                    else:
                        formatted_messages.append(HumanMessage(content=content))
                else:
                    formatted_messages.append(msg)
            
            # Stream from LLM
            async for chunk in self.llm.astream(formatted_messages):
                if hasattr(chunk, 'content'):
                    yield chunk.content
                else:
                    yield str(chunk)
                    
        except Exception as e:
            print(f"Error in streaming: {e}")
            yield f"Error: {str(e)}"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        return {
            "provider": self.provider.value,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
    
    @staticmethod
    def create_fallback_chain(primary_wrapper: 'LLMWrapper', fallback_wrapper: 'LLMWrapper'):
        """
        Create a fallback chain between two LLM wrappers
        
        Args:
            primary_wrapper: Primary LLM to try first
            fallback_wrapper: Fallback LLM if primary fails
            
        Returns:
            Function that tries primary then fallback
        """
        def invoke_with_fallback(messages: list) -> str:
            try:
                return primary_wrapper.invoke(messages)
            except Exception as e:
                print(f"Primary LLM failed: {e}. Trying fallback...")
                try:
                    return fallback_wrapper.invoke(messages)
                except Exception as e2:
                    print(f"Fallback LLM also failed: {e2}")
                    return "Error: Both primary and fallback LLMs failed."
        
        return invoke_with_fallback


# Factory function for easy LLM creation
def create_llm(
    provider: str = "openai",
    model_name: Optional[str] = None,
    temperature: float = 0.7,
    **kwargs
) -> LLMWrapper:
    """
    Factory function to create LLM wrapper
    
    Args:
        provider: 'openai' or 'ollama'
        model_name: Model name
        temperature: Temperature value
        **kwargs: Additional arguments
        
    Returns:
        LLMWrapper instance
    """
    return LLMWrapper(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        **kwargs
    )


# Example usage
if __name__ == "__main__":
    # Test OpenAI
    try:
        llm_openai = create_llm(provider="openai", model_name="gpt-3.5-turbo")
        messages = [
            {"role": "system", "content": "You are a helpful astrology assistant."},
            {"role": "user", "content": "What is the sun sign for someone born on August 20?"}
        ]
        response = llm_openai.invoke(messages)
        print("OpenAI Response:", response)
    except Exception as e:
        print(f"OpenAI test failed: {e}")
    
    # Test Ollama
    try:
        llm_ollama = create_llm(provider="ollama", model_name="llama3.1")
        response = llm_ollama.invoke(messages)
        print("Ollama Response:", response)
    except Exception as e:
        print(f"Ollama test failed: {e}")