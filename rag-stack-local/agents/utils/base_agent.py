from ollama import Client, ChatResponse
import json
from .open_ai_schema import OpenAiSchema
from langfuse import observe

# Base Agent class
class BaseAgent:
    def __init__(self, name, host, model):
        self.client = Client(host=host)
        self.model = model
        self.host = host
        self.name = name
    
    @observe()
    def get_model_info(self):
        return OpenAiSchema.model_info(self.name)
    
    @observe()
    def format_to_openai_response(self, response: ChatResponse):
        return OpenAiSchema.format_ollama_chat_response_to_openai_response(response, self.name)
    
    @observe()
    def final_answer_to_openai_response(self, response: ChatResponse):
        return OpenAiSchema.final_openai_response(response, self.name)
    
    @observe()
    def opan_ai_stream(self, ollama_stream):
        return OpenAiSchema.format_ollama_to_openai_streaming(ollama_stream, self.name)
    
    @observe()
    def opan_ai_stream_to_openai_stream(self, ollama_stream):
        return OpenAiSchema.format_openai_to_openai_streaming(ollama_stream, self.name)
        
    
        
        
    