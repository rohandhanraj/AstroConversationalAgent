"""
LangGraph Agent Module
Implements the conversational agent using LangGraph
"""
from typing import TypedDict, Annotated, List, Dict, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import json

from app.llm_wrapper import LLMWrapper
from app.retrieval import RetrievalEngine
from app.translation import TranslationService
from app.memory import ConversationMemory


class AgentState(TypedDict):
    """State for the agent graph"""
    messages: List[Dict]
    user_profile: Dict
    query: str
    retrieved_context: List[str]
    response: str
    context_used: List[str]
    language: str
    confidence_score: float
    error: Optional[str]


class AstroConversationalAgent:
    """
    Conversational agent for astrology insights using LangGraph
    """
    
    def __init__(
        self,
        llm_wrapper: LLMWrapper,
        retrieval_engine: RetrievalEngine,
        translation_service: Optional[TranslationService] = None,
        enable_hallucination_filter: bool = True
    ):
        """
        Initialize the agent
        
        Args:
            llm_wrapper: LLM wrapper instance
            retrieval_engine: Retrieval engine instance
            translation_service: Optional translation service
            enable_hallucination_filter: Enable hallucination filtering
        """
        self.llm = llm_wrapper
        self.retrieval = retrieval_engine
        self.translator = translation_service or TranslationService()
        self.enable_hallucination_filter = enable_hallucination_filter
        
        # Build the agent graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("retrieve", self._retrieve_context)
        workflow.add_node("generate", self._generate_response)
        workflow.add_node("filter", self._filter_response)
        workflow.add_node("translate", self._translate_response)
        
        # Add edges
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        
        # Conditional edge based on filtering
        if self.enable_hallucination_filter:
            workflow.add_edge("generate", "filter")
            workflow.add_edge("filter", "translate")
        else:
            workflow.add_edge("generate", "translate")
        
        workflow.add_edge("translate", END)
        
        return workflow.compile()
    
    def _retrieve_context(self, state: AgentState) -> AgentState:
        """
        Retrieve relevant context from knowledge base
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with retrieved context
        """
        try:
            query = state["query"]
            user_profile = state["user_profile"]
            
            # Enhance query with user profile info
            enhanced_query = f"{query} {user_profile.get('sun_sign', '')} zodiac"
            
            # Retrieve context
            results = self.retrieval.retrieve(enhanced_query, k=3)
            
            context_list = []
            context_sources = []
            
            for doc, score in results:
                context_list.append(doc.page_content)
                source = doc.metadata.get("source", "unknown")
                context_sources.append(source)
            
            state["retrieved_context"] = context_list
            state["context_used"] = list(set(context_sources))
            
        except Exception as e:
            print(f"Retrieval error: {e}")
            state["retrieved_context"] = []
            state["context_used"] = []
            state["error"] = f"Retrieval error: {str(e)}"
        
        return state
    
    def _generate_response(self, state: AgentState) -> AgentState:
        """
        Generate response using LLM
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with generated response
        """
        try:
            query = state["query"]
            user_profile = state["user_profile"]
            context = state.get("retrieved_context", [])
            messages = state.get("messages", [])
            
            # Build context string
            context_str = "\n\n".join(context) if context else "No specific context found."
            
            # Build conversation history string
            history_str = ""
            if messages:
                recent_messages = messages[-6:]  # Last 3 turns
                for msg in recent_messages:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    history_str += f"{role.capitalize()}: {content}\n"
            
            # Create system prompt
            system_prompt = """You are an expert Vedic astrology consultant. 
You provide personalized, compassionate, and insightful guidance based on astrological principles.

Guidelines:
- Use the provided context and user profile to give personalized responses
- Be empathetic and supportive in your tone
- Reference specific astrological concepts when relevant (planets, transits, zodiac traits)
- Keep responses concise but meaningful (2-4 paragraphs)
- If information is not in context, use general astrological knowledge
- Never make harmful predictions or create unnecessary anxiety
- Focus on guidance and self-empowerment

Always maintain a balance between astrological insights and practical wisdom."""
            
            # Create user prompt
            user_prompt = f"""Context from Knowledge Base:
{context_str}

User Profile:
- Name: {user_profile.get('name', 'User')}
- Sun Sign: {user_profile.get('sun_sign', 'Unknown')}
- Moon Sign: {user_profile.get('moon_sign', 'Unknown')}
- Ascendant: {user_profile.get('ascendant', 'Unknown')}
- Nakshatra: {user_profile.get('nakshatra', 'Unknown')}
- Age: {user_profile.get('age', 'Unknown')}

Recent Conversation:
{history_str}

Current Question: {query}

Please provide a personalized astrological response based on the user's profile and the context provided."""
            
            # Generate response
            messages_to_llm = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = self.llm.invoke(messages_to_llm)
            
            state["response"] = response
            state["confidence_score"] = self._calculate_confidence(context, response)
            
        except Exception as e:
            print(f"Generation error: {e}")
            state["response"] = "I apologize, but I encountered an error generating your response. Please try again."
            state["confidence_score"] = 0.0
            state["error"] = f"Generation error: {str(e)}"
        
        return state
    
    async def _generate_response_stream(self, state: AgentState):
        """
        Generate response using LLM with streaming (for async graph streaming)
        
        Args:
            state: Current agent state
            
        Yields:
            Updated state chunks with streaming response
        """
        try:
            query = state["query"]
            user_profile = state["user_profile"]
            context = state.get("retrieved_context", [])
            messages = state.get("messages", [])
            
            # Build context and prompts (same as non-streaming)
            context_str = "\n\n".join(context) if context else "No specific context found."
            
            history_str = ""
            if messages:
                recent_messages = messages[-6:]
                for msg in recent_messages:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    history_str += f"{role.capitalize()}: {content}\n"
            
            system_prompt = """You are an expert Vedic astrology consultant. 
You provide personalized, compassionate, and insightful guidance based on astrological principles.

Guidelines:
- Use the provided context and user profile to give personalized responses
- Be empathetic and supportive in your tone
- Reference specific astrological concepts when relevant
- Keep responses concise but meaningful (2-4 paragraphs)
- Never make harmful predictions or create unnecessary anxiety
- Focus on guidance and self-empowerment"""
            
            user_prompt = f"""Context from Knowledge Base:
{context_str}

User Profile:
- Name: {user_profile.get('name', 'User')}
- Sun Sign: {user_profile.get('sun_sign', 'Unknown')}
- Moon Sign: {user_profile.get('moon_sign', 'Unknown')}
- Age: {user_profile.get('age', 'Unknown')}

Recent Conversation:
{history_str}

Current Question: {query}

Please provide a personalized astrological response."""
            
            messages_to_llm = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Stream LLM response
            full_response = ""
            async for chunk in self.llm.astream(messages_to_llm):
                full_response += chunk
                # Update state incrementally
                state["response"] = full_response
                yield state
            
            # Final state update
            state["confidence_score"] = self._calculate_confidence(context, full_response)
            yield state
            
        except Exception as e:
            print(f"Streaming generation error: {e}")
            state["response"] = "I apologize, but I encountered an error. Please try again."
            state["confidence_score"] = 0.0
            state["error"] = f"Generation error: {str(e)}"
            yield state
    
    def _filter_response(self, state: AgentState) -> AgentState:
        """
        Filter response for hallucinations and inappropriate content
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with filtered response
        """
        try:
            response = state["response"]
            
            # Simple filtering rules
            toxic_keywords = ["die", "death", "doom", "curse", "evil", "disaster"]
            overly_confident = ["definitely will", "guaranteed", "100% certain", "absolutely will"]
            
            # Check for toxic content
            response_lower = response.lower()
            has_toxic = any(keyword in response_lower for keyword in toxic_keywords)
            has_overconfident = any(phrase in response_lower for phrase in overly_confident)
            
            if has_toxic or has_overconfident:
                # Add disclaimer
                filtered_response = response + "\n\n*Note: Astrological insights are for guidance only and should not be taken as absolute predictions.*"
                state["response"] = filtered_response
                state["confidence_score"] *= 0.8  # Reduce confidence
            
        except Exception as e:
            print(f"Filtering error: {e}")
        
        return state
    
    def _translate_response(self, state: AgentState) -> AgentState:
        """
        Translate response if needed
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with translated response
        """
        try:
            language = state.get("language", "en")
            
            if language == "hi":
                response = state["response"]
                translated = self.translator.translate_to_hindi(response)
                state["response"] = translated
                
        except Exception as e:
            print(f"Translation error: {e}")
            state["error"] = f"Translation error: {str(e)}"
        
        return state
    
    def _calculate_confidence(self, context: List[str], response: str) -> float:
        """
        Calculate confidence score for the response
        
        Args:
            context: Retrieved context
            response: Generated response
            
        Returns:
            Confidence score (0-1)
        """
        # Simple heuristic: based on context availability and response length
        if not context:
            base_score = 0.5
        else:
            base_score = 0.8
        
        # Adjust based on response length
        if len(response) < 50:
            return base_score * 0.6
        elif len(response) < 200:
            return base_score * 0.8
        else:
            return base_score
    
    def process_query(
        self,
        query: str,
        user_profile: Dict,
        conversation_history: List[Dict],
        language: str = "en"
    ) -> Dict:
        """
        Process a user query through the agent graph
        
        Args:
            query: User's question
            user_profile: User's astrological profile
            conversation_history: Previous messages
            language: Target language ('en' or 'hi')
            
        Returns:
            Response dictionary
        """
        # Initialize state
        initial_state: AgentState = {
            "messages": conversation_history,
            "user_profile": user_profile,
            "query": query,
            "retrieved_context": [],
            "response": "",
            "context_used": [],
            "language": language,
            "confidence_score": 0.0,
            "error": None
        }
        
        # Run the graph
        final_state = self.graph.invoke(initial_state)
        
        # Build response
        return {
            "response": final_state["response"],
            "context_used": final_state["context_used"],
            "zodiac": user_profile.get("sun_sign", "Unknown"),
            "confidence_score": final_state["confidence_score"],
            "error": final_state.get("error")
        }
    
    async def process_query_stream(
        self,
        query: str,
        user_profile: Dict,
        conversation_history: List[Dict],
        language: str = "en"
    ):
        """
        Process query with streaming response using LangGraph streaming
        
        Args:
            query: User's question
            user_profile: User's astrological profile
            conversation_history: Previous messages
            language: Target language ('en' or 'hi')
            
        Yields:
            Response chunks and metadata
        """
        try:
            # Initialize state
            initial_state: AgentState = {
                "messages": conversation_history,
                "user_profile": user_profile,
                "query": query,
                "retrieved_context": [],
                "response": "",
                "context_used": [],
                "language": language,
                "confidence_score": 0.0,
                "error": None
            }
            
            context_used = []
            confidence_score = 0.0
            
            # Track which nodes have completed
            nodes_completed = set()
            
            # Stream through the graph with multiple modes
            async for event in self.graph.astream(
                initial_state,
                stream_mode=["updates", "messages"]
            ):
                event_type, data = event
                
                if event_type == "updates":
                    # Node execution updates
                    for node_name, node_state in data.items():
                        if node_name not in nodes_completed:
                            # Yield node start
                            yield {
                                "type": "node_start",
                                "node": node_name,
                                "message": self._get_node_message(node_name)
                            }
                            nodes_completed.add(node_name)
                        
                        # Update metadata from state
                        if "context_used" in node_state:
                            context_used = node_state["context_used"]
                        if "confidence_score" in node_state:
                            confidence_score = node_state["confidence_score"]
                
                elif event_type == "messages":
                    # Message chunks from LLM
                    message, metadata = data
                    
                    # Check if this is from the generate node (before translation)
                    if hasattr(message, 'content') and message.content:
                        # Only stream if we're in English mode or before translation
                        if language == "en":
                            yield {
                                "type": "chunk",
                                "content": message.content
                            }
                        else:
                            # For Hindi, we'll wait for the translation node to complete
                            # Just store the content but don't stream yet
                            pass
            
            # After graph completes, get final state
            final_state = await self.graph.ainvoke(initial_state)
            final_response = final_state.get("response", "")
            
            # For Hindi, stream the translated response all at once
            if language == "hi" and final_response:
                yield {
                    "type": "chunk",
                    "content": final_response
                }
            
            # Send completion with final metadata
            yield {
                "type": "done",
                "context_used": final_state.get("context_used", context_used),
                "confidence_score": final_state.get("confidence_score", confidence_score),
                "zodiac": user_profile.get("sun_sign", "Unknown"),
                "final_response": final_response
            }
            
        except Exception as e:
            print(f"Error in streaming query: {e}")
            import traceback
            traceback.print_exc()
            yield {
                "type": "error",
                "message": str(e)
            }
    
    def _get_node_message(self, node_name: str) -> str:
        """Get user-friendly message for node execution"""
        messages = {
            "retrieve": "🔍 Retrieving astrological context...",
            "generate": "✨ Generating personalized response...",
            "filter": "🔒 Applying safety filters...",
            "translate": "🌐 Translating to Hindi..."
        }
        return messages.get(node_name, f"Processing {node_name}...")


# Example usage
if __name__ == "__main__":
    from app.llm_wrapper import create_llm
    
    # Initialize components
    llm = create_llm(provider="openai", model_name="gpt-3.5-turbo", temperature=0.7)
    
    retrieval = RetrievalEngine(
        data_dir="../data",
        vector_store_type="faiss"
    )
    retrieval.build_vector_store()
    
    # Create agent
    agent = AstroConversationalAgent(llm, retrieval)
    
    # Test query
    user_profile = {
        "name": "Ritika",
        "sun_sign": "Leo",
        "moon_sign": "Libra",
        "ascendant": "Sagittarius",
        "age": 29
    }
    
    response = agent.process_query(
        query="How will my month be in career?",
        user_profile=user_profile,
        conversation_history=[],
        language="en"
    )
    
    print(json.dumps(response, indent=2))