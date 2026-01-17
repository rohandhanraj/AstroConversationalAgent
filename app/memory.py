"""
Memory Management Module
Handles conversation state and session memory
"""
from typing import Dict, List, Optional
from datetime import datetime
from collections import defaultdict
import json


class ConversationMemory:
    """
    Manages conversation history for a single session
    """
    
    def __init__(self, session_id: str, max_history: int = 20):
        """
        Initialize conversation memory
        
        Args:
            session_id: Unique session identifier
            max_history: Maximum number of messages to keep in history
        """
        self.session_id = session_id
        self.max_history = max_history
        self.messages: List[Dict] = []
        self.user_profile: Optional[Dict] = None
        self.created_at = datetime.now()
        self.last_updated = datetime.now()
        self.metadata: Dict = {}
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """
        Add a message to conversation history
        
        Args:
            role: Message role ('user', 'assistant', 'system')
            content: Message content
            metadata: Optional metadata
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        }
        
        if metadata:
            message["metadata"] = metadata
        
        self.messages.append(message)
        self.last_updated = datetime.now()
        
        # Trim history if too long
        if len(self.messages) > self.max_history:
            # Keep system messages and trim user/assistant messages
            system_messages = [m for m in self.messages if m["role"] == "system"]
            other_messages = [m for m in self.messages if m["role"] != "system"]
            
            # Keep last N messages
            keep_count = self.max_history - len(system_messages)
            trimmed_messages = other_messages[-keep_count:]
            
            self.messages = system_messages + trimmed_messages
    
    def get_messages(self, last_n: Optional[int] = None) -> List[Dict]:
        """
        Get conversation messages
        
        Args:
            last_n: Optional number of last messages to return
            
        Returns:
            List of message dictionaries
        """
        if last_n:
            return self.messages[-last_n:]
        return self.messages
    
    def get_context_string(self, last_n: Optional[int] = None) -> str:
        """
        Get conversation history as a formatted string
        
        Args:
            last_n: Optional number of last messages to include
            
        Returns:
            Formatted conversation history
        """
        messages = self.get_messages(last_n)
        context_parts = []
        
        for msg in messages:
            role = msg["role"].capitalize()
            content = msg["content"]
            context_parts.append(f"{role}: {content}")
        
        return "\n".join(context_parts)
    
    def set_user_profile(self, profile: Dict):
        """Set or update user profile"""
        self.user_profile = profile
        self.last_updated = datetime.now()
    
    def get_user_profile(self) -> Optional[Dict]:
        """Get user profile"""
        return self.user_profile
    
    def add_metadata(self, key: str, value: any):
        """Add metadata to session"""
        self.metadata[key] = value
        self.last_updated = datetime.now()
    
    def get_metadata(self, key: str) -> Optional[any]:
        """Get metadata value"""
        return self.metadata.get(key)
    
    def clear(self):
        """Clear conversation history (keeping profile)"""
        self.messages = []
        self.last_updated = datetime.now()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "session_id": self.session_id,
            "messages": self.messages,
            "user_profile": self.user_profile,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ConversationMemory':
        """Create from dictionary"""
        memory = cls(data["session_id"])
        memory.messages = data.get("messages", [])
        memory.user_profile = data.get("user_profile")
        memory.created_at = datetime.fromisoformat(data["created_at"])
        memory.last_updated = datetime.fromisoformat(data["last_updated"])
        memory.metadata = data.get("metadata", {})
        return memory


class SessionManager:
    """
    Manages multiple conversation sessions
    """
    
    def __init__(self, max_sessions: int = 1000):
        """
        Initialize session manager
        
        Args:
            max_sessions: Maximum number of sessions to keep in memory
        """
        self.sessions: Dict[str, ConversationMemory] = {}
        self.max_sessions = max_sessions
    
    def get_session(self, session_id: str) -> ConversationMemory:
        """
        Get or create a session
        
        Args:
            session_id: Session identifier
            
        Returns:
            ConversationMemory instance
        """
        if session_id not in self.sessions:
            # Check if we need to clean up old sessions
            if len(self.sessions) >= self.max_sessions:
                self._cleanup_old_sessions()
            
            self.sessions[session_id] = ConversationMemory(session_id)
        
        return self.sessions[session_id]
    
    def delete_session(self, session_id: str):
        """Delete a session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
    
    def _cleanup_old_sessions(self):
        """Clean up oldest sessions when limit reached"""
        if not self.sessions:
            return
        
        # Sort by last updated time
        sorted_sessions = sorted(
            self.sessions.items(),
            key=lambda x: x[1].last_updated
        )
        
        # Remove oldest 10%
        remove_count = max(1, len(self.sessions) // 10)
        for session_id, _ in sorted_sessions[:remove_count]:
            del self.sessions[session_id]
    
    def get_active_sessions_count(self) -> int:
        """Get count of active sessions"""
        return len(self.sessions)
    
    def save_sessions(self, filepath: str):
        """
        Save all sessions to file
        
        Args:
            filepath: Path to save file
        """
        data = {
            session_id: memory.to_dict()
            for session_id, memory in self.sessions.items()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_sessions(self, filepath: str):
        """
        Load sessions from file
        
        Args:
            filepath: Path to load file
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.sessions = {
            session_id: ConversationMemory.from_dict(session_data)
            for session_id, session_data in data.items()
        }


# Global session manager instance
_session_manager = None


def get_session_manager() -> SessionManager:
    """Get global session manager instance"""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager


# Example usage
if __name__ == "__main__":
    # Create session manager
    manager = SessionManager()
    
    # Get a session
    session = manager.get_session("test-123")
    
    # Add user profile
    profile = {
        "name": "Ritika",
        "sun_sign": "Leo",
        "birth_date": "1995-08-20"
    }
    session.set_user_profile(profile)
    
    # Add messages
    session.add_message("user", "What are my career strengths?")
    session.add_message("assistant", "As a Leo, you excel in leadership roles...")
    session.add_message("user", "How will this month be for me?")
    
    # Get context
    print("Conversation History:")
    print(session.get_context_string())
    
    print(f"\nTotal messages: {len(session.get_messages())}")
    print(f"User profile: {session.get_user_profile()}")
