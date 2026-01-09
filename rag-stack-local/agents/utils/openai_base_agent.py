from .base_agent import BaseAgent
from openai import OpenAI
from langfuse import observe

class OpenAiBaseAgent(BaseAgent):
    def __init__(self, name, host, api_key, model):
        self.client = OpenAI(base_url=host, api_key=api_key)
        self.model = model
        self.name = name
        
    @observe()
    def chat(self, messages: list, stream:bool = False):
        return self.client.chat.completions.create(model=self.model, messages=messages, stream=stream)