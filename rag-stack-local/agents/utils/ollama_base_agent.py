from .base_agent import BaseAgent
from ollama import Client


class OllamaBaseAgent(BaseAgent):
    def __init__(self, name, host, model):
        self.client = Client(host=host)
        self.model = model
        self.name = name

    def chat(self, messages: list, stream: bool = False, tools: list = []):
        return self.client.chat(model=self.model, messages=messages, stream=stream, tools=tools)
