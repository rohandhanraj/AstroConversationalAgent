"""
MongoDB Logger Module
Handles logging of requests and responses to MongoDB
"""
import os
from datetime import datetime
from typing import Dict, Optional, Any
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import logging

logger = logging.getLogger(__name__)


class MongoDBLogger:
    """
    Logger for storing requests and responses in MongoDB
    """
    
    def __init__(self, connection_string: Optional[str] = None):
        """
        Initialize MongoDB logger
        
        Args:
            connection_string: MongoDB connection string (Atlas URL)
        """
        self.connection_string = connection_string or os.getenv(
            "MONGODB_URI",
            "mongodb://localhost:27017/"  # Fallback to local
        )
        self.database_name = os.getenv("MONGODB_DATABASE", "astro_agent")
        self.collection_name = "conversation_logs"
        
        self.client = None
        self.db = None
        self.collection = None
        self.connected = False
        
        # Try to connect
        self._connect()
    
    def _connect(self):
        """Connect to MongoDB"""
        try:
            self.client = MongoClient(
                self.connection_string,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000
            )
            # Test connection
            self.client.server_info()
            
            self.db = self.client[self.database_name]
            self.collection = self.db[self.collection_name]
            self.connected = True
            
            # Create indexes for better query performance
            self.collection.create_index("session_id")
            self.collection.create_index("timestamp")
            self.collection.create_index([("session_id", 1), ("timestamp", -1)])
            
            logger.info(f"✓ MongoDB connected: {self.database_name}.{self.collection_name}")
            
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.warning(f"⚠ MongoDB connection failed: {e}")
            logger.warning("Continuing without MongoDB logging...")
            self.connected = False
        except Exception as e:
            logger.error(f"✗ MongoDB initialization error: {e}")
            self.connected = False
    
    def log_request_response(
        self,
        session_id: str,
        request_data: Dict[str, Any],
        response_data: Dict[str, Any],
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Log a request-response pair to MongoDB
        
        Args:
            session_id: Session identifier
            request_data: Request information
            response_data: Response information
            metadata: Optional additional metadata
            
        Returns:
            True if logged successfully, False otherwise
        """
        if not self.connected:
            return False
        
        try:
            log_entry = {
                "session_id": session_id,
                "timestamp": datetime.utcnow(),
                "request": {
                    "message": request_data.get("message"),
                    "user_profile": request_data.get("user_profile"),
                    "language": request_data.get("language", "en")
                },
                "response": {
                    "text": response_data.get("response"),
                    "context_used": response_data.get("context_used", []),
                    "confidence_score": response_data.get("confidence_score"),
                    "zodiac": response_data.get("zodiac")
                },
                "metadata": metadata or {}
            }
            
            result = self.collection.insert_one(log_entry)
            logger.debug(f"Logged conversation: {result.inserted_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to log to MongoDB: {e}")
            return False
    
    def get_session_history(
        self,
        session_id: str,
        limit: int = 50
    ) -> list:
        """
        Retrieve conversation history for a session
        
        Args:
            session_id: Session identifier
            limit: Maximum number of entries to retrieve
            
        Returns:
            List of conversation entries
        """
        if not self.connected:
            return []
        
        try:
            cursor = self.collection.find(
                {"session_id": session_id}
            ).sort("timestamp", -1).limit(limit)
            
            return list(cursor)
            
        except Exception as e:
            logger.error(f"Failed to retrieve session history: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get usage statistics from MongoDB
        
        Returns:
            Dictionary with statistics
        """
        if not self.connected:
            return {"connected": False}
        
        try:
            total_conversations = self.collection.count_documents({})
            unique_sessions = len(self.collection.distinct("session_id"))
            
            # Get recent activity (last 24 hours)
            from datetime import timedelta
            day_ago = datetime.utcnow() - timedelta(days=1)
            recent_count = self.collection.count_documents(
                {"timestamp": {"$gte": day_ago}}
            )
            
            return {
                "connected": True,
                "total_conversations": total_conversations,
                "unique_sessions": unique_sessions,
                "last_24h_conversations": recent_count,
                "database": self.database_name,
                "collection": self.collection_name
            }
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {"connected": True, "error": str(e)}
    
    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")


# Global MongoDB logger instance
_mongodb_logger = None


def get_mongodb_logger() -> MongoDBLogger:
    """Get or create MongoDB logger instance"""
    global _mongodb_logger
    if _mongodb_logger is None:
        _mongodb_logger = MongoDBLogger()
    return _mongodb_logger


# Example usage
if __name__ == "__main__":
    # Test MongoDB logger
    mongo_logger = MongoDBLogger()
    
    if mongo_logger.connected:
        print("✓ MongoDB connected successfully")
        
        # Test logging
        test_logged = mongo_logger.log_request_response(
            session_id="test-123",
            request_data={
                "message": "What is my career strength?",
                "user_profile": {"name": "Test", "sun_sign": "Leo"}
            },
            response_data={
                "response": "As a Leo, you excel in leadership...",
                "context_used": ["leo_traits"],
                "confidence_score": 0.85,
                "zodiac": "Leo"
            }
        )
        
        if test_logged:
            print("✓ Test log entry created")
        
        # Get statistics
        stats = mongo_logger.get_statistics()
        print(f"Statistics: {stats}")
        
        mongo_logger.close()
    else:
        print("✗ MongoDB not connected")
