"""
Translation Utility Module
Handles English to Hindi translation for responses
"""
from typing import Optional
from deep_translator import GoogleTranslator


class TranslationService:
    """Service for translating text between languages"""
    
    def __init__(self):
        """Initialize translation service"""
        self.translator_en_to_hi = GoogleTranslator(source='en', target='hi')
        self.translator_hi_to_en = GoogleTranslator(source='hi', target='en')
    
    def translate_to_hindi(self, text: str) -> str:
        """
        Translate English text to Hindi
        
        Args:
            text: English text to translate
            
        Returns:
            Hindi translation
        """
        try:
            # Split into chunks if text is too long (Google Translate has limits)
            max_length = 4500
            if len(text) <= max_length:
                return self.translator_en_to_hi.translate(text)
            
            # Split and translate in chunks
            chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
            translated_chunks = []
            
            for chunk in chunks:
                translated = self.translator_en_to_hi.translate(chunk)
                translated_chunks.append(translated)
            
            return ' '.join(translated_chunks)
            
        except Exception as e:
            print(f"Translation error: {e}")
            return text  # Return original text if translation fails
    
    def translate_to_english(self, text: str) -> str:
        """
        Translate Hindi text to English
        
        Args:
            text: Hindi text to translate
            
        Returns:
            English translation
        """
        try:
            max_length = 4500
            if len(text) <= max_length:
                return self.translator_hi_to_en.translate(text)
            
            # Split and translate in chunks
            chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
            translated_chunks = []
            
            for chunk in chunks:
                translated = self.translator_hi_to_en.translate(chunk)
                translated_chunks.append(translated)
            
            return ' '.join(translated_chunks)
            
        except Exception as e:
            print(f"Translation error: {e}")
            return text
    
    def detect_language(self, text: str) -> str:
        """
        Detect language of text (simple heuristic)
        
        Args:
            text: Text to analyze
            
        Returns:
            'hi' for Hindi, 'en' for English
        """
        # Simple detection: check for Devanagari characters
        hindi_chars = sum(1 for char in text if '\u0900' <= char <= '\u097F')
        total_chars = len([c for c in text if c.isalpha()])
        
        if total_chars > 0 and hindi_chars / total_chars > 0.3:
            return 'hi'
        return 'en'


class MultilingualLLMWrapper:
    """
    Wrapper for LLM with automatic translation support
    """
    
    def __init__(self, llm_wrapper, translation_service: Optional[TranslationService] = None):
        """
        Initialize multilingual wrapper
        
        Args:
            llm_wrapper: Base LLM wrapper
            translation_service: Translation service instance
        """
        self.llm = llm_wrapper
        self.translator = translation_service or TranslationService()
    
    def invoke(self, messages: list, target_language: str = "en") -> str:
        """
        Invoke LLM with automatic translation
        
        Args:
            messages: List of messages
            target_language: Target language ('en' or 'hi')
            
        Returns:
            Response in target language
        """
        # Get response in English
        response = self.llm.invoke(messages)
        
        # Translate if needed
        if target_language == "hi":
            response = self.translator.translate_to_hindi(response)
        
        return response
    
    async def ainvoke(self, messages: list, target_language: str = "en") -> str:
        """
        Async invoke LLM with automatic translation
        
        Args:
            messages: List of messages
            target_language: Target language ('en' or 'hi')
            
        Returns:
            Response in target language
        """
        # Get response in English
        response = await self.llm.ainvoke(messages)
        
        # Translate if needed
        if target_language == "hi":
            response = self.translator.translate_to_hindi(response)
        
        return response


# Hindi-specific prompt templates
HINDI_SYSTEM_PROMPT = """आप एक सहायक ज्योतिष सलाहकार हैं। आप वैदिक ज्योतिष के सिद्धांतों का उपयोग करते हुए व्यक्तिगत और सटीक मार्गदर्शन प्रदान करते हैं।"""

HINDI_PROMPT_TEMPLATE = """
संदर्भ: {context}

उपयोगकर्ता प्रोफाइल:
नाम: {name}
राशि: {sun_sign}
चंद्र राशि: {moon_sign}
लग्न: {ascendant}

प्रश्न: {question}

कृपया उपरोक्त जानकारी और संदर्भ के आधार पर एक व्यक्तिगत और सहायक उत्तर प्रदान करें।
"""


# Example usage
if __name__ == "__main__":
    translator = TranslationService()
    
    # Test English to Hindi
    english_text = "Leo individuals are natural leaders and excel in creative fields. They should focus on building their confidence and pursuing their passions."
    hindi_text = translator.translate_to_hindi(english_text)
    print(f"English: {english_text}")
    print(f"Hindi: {hindi_text}")
    
    # Test language detection
    print(f"\nDetected language: {translator.detect_language(hindi_text)}")
    
    # Test Hindi to English
    back_to_english = translator.translate_to_english(hindi_text)
    print(f"Back to English: {back_to_english}")
