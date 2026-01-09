import json
from ollama import ChatResponse

class OpenAiSchema:
    
    @staticmethod
    def model_info(model_name):
        return {
         "object": "list",
        "data": [
            {
            "id": model_name,
            "object": "model",
            "owned_by": "organization-owner"
            },
        ],
        "object": "list"
        }
    
    @staticmethod
    def format_ollama_chat_response_to_openai_response(ollama_response: ChatResponse, model_id):
        tool_calls = []
        if ollama_response.message.tool_calls:
            for tool_call in ollama_response.message.tool_calls:
                tool_calls.append({
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        }
                    }
                )
                
        response_content = ollama_response.message.content
        if tool_calls:
            response_content += f"\n### Tool calls:\n```json{tool_calls}\n```"
        
        formatted_response = {
            "id": "12345",
            "object": "chat.completion",
            "created":  ollama_response.created_at,
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_content,
                        "tool_calls": tool_calls
                    },
                    "finish_reason": "stop"
                }
            ],
        }
        return json.dumps(formatted_response, indent=4)
    
    @staticmethod  
    def final_openai_response(response:str, model_id):
        tool_calls = []
        
        formatted_response = {
            "id": "12345",
            "object": "chat.completion",
            # "created":  ollama_response.created_at,
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response,
                        "tool_calls": tool_calls
                    },
                    "finish_reason": "stop"
                }
            ],
        }
        return json.dumps(formatted_response, indent=4)

        
    def format_ollama_to_openai_streaming(ollama_stream, model_id):
        """
        Format the Ollama response into OpenAI streaming format.
        """
        
        for chunk in ollama_stream:
            response_chunk = json.dumps({
                "id": "12345",
                "object": "chat.completion.chunk",
                "model": model_id,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant", 
                            "content": chunk["message"]["content"]
                        },
                        "finish_reason": None
                    }
                ]
            })
            yield f"data: {response_chunk}\n\n"

        # Send final completion signal
        yield f"data: {json.dumps({'id': '12345', 'object': 'chat.completion.chunk', 'model': model_id, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
        
    def format_openai_to_openai_streaming(ollama_stream, model_id):
        """
        Format the Ollama response into OpenAI streaming format.
        """
        
        for chunk in ollama_stream:
            response_chunk = json.dumps({
                "id": "12345",
                "object": "chat.completion.chunk",
                "model": model_id,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant", 
                            "content": chunk.choices[0].delta.content
                        },
                        "finish_reason": None
                    }
                ]
            })
            yield f"data: {response_chunk}\n\n"

        # Send final completion signal
        yield f"data: {json.dumps({'id': '12345', 'object': 'chat.completion.chunk', 'model': model_id, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"