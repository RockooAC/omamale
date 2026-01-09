from pydantic import BaseModel
import json


class AgentAction(BaseModel):
    tool_name: str = ""
    tool_input: dict = {}
    tool_output: str | None = None

    @classmethod
    def from_ollama(cls, ollama_response):
        try:
            output = ollama_response.message.tool_calls[0]
            return cls(
                tool_name=output.function.name,
                tool_input=output.function.arguments
            )
        except Exception as e:
            print(f"Error parsing ollama response:\n{ollama_response}\n")
            raise e

    def __str__(self):
        text = f"Tool: {self.tool_name}\nInput: {self.tool_input}"
        if self.tool_output is not None:
            text += f"\nOutput: {self.tool_output}"
        return text
    
    def to_message(self):
        # create assistant "input" message
        assistant_content = f"call: {self.tool_name} with parameters: {self.tool_input}"
        assistant_message = {"role": "assistant", "content": assistant_content}
        # create user "response" message
        tool_message = {"role": "tool", "content": self.tool_output}
        return [assistant_message, tool_message]
    
    def to_json(self):
        return {
            "tool_name": self.tool_name,
            "tool_input": self.tool_input,
            "tool_output": self.tool_output
        }
        
    def from_json(self, obj):
        self.tool_name = obj["tool_name"]
        self.tool_input = obj["tool_input"]
        self.tool_output = obj["tool_output"]
    
    