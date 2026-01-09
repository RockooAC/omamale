import sys
import os

if not os.path.exists('/.dockerenv'):  # local debugging
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.env import Env
from langfuse import observe, get_client
from flask import Flask, request, jsonify, Response
from utils import OllamaBaseAgent, logger, getSessionID, setSessionIDHeader


logger = logger.setup_logger("final-answer-agent")


# Final Answer Agent
class FinalAnswerAgent(OllamaBaseAgent):
    def __init__(self, name, host, model):
        super().__init__(name, host, model)
        self.system_prompt = """
        You are the Final Answer Agent, responsible for synthesizing information from specialized tools and your internal knowledge to deliver the most accurate and comprehensive response to the user.

        **Responsibilities:**  
        - Aggregate insights from various tools provided by the Oracle to create a complete, accurate, and user-friendly answer.  
        - Fill any information gaps using your internal knowledge, ensuring a fully rounded response.  
        - Craft responses that are clear, concise, and comprehensive.  
          
        **Input:**  
        The Oracle will provide you with all relevant information gathered from the specialized tools, along with the original user query.

        **Output:**  
        - Your response should be:  
            - **Clear and Concise:** Easy to understand for users, including those with minimal technical knowledge.  
            - **Complete:** Address every aspect of the query thoroughly, leaving no unresolved questions.  
            - **Actionable:** Provide clear next steps or practical solutions where applicable.  
        """ 

    @observe()
    def execute(self, messages) -> str:
        mess = (
            {"role": "system", "content": f"{self.system_prompt}"},
            *messages
        )
        response = self.chat(messages=mess)
        return response
    
    @observe()
    def stream(self, query: list[dict]):
        messages = (
            {"role": "system", "content": self.system_prompt},
            *query
        )
        for chunk in  self.chat(messages=messages, stream=True):
            yield chunk


app = Flask(__name__)
agent = FinalAnswerAgent('final-answer-agent', Env("OLLAMA_URL").get(), Env("OLLAMA_MODEL").get())


@app.route('/models', methods=['GET'])
def get_model_info():
    return jsonify(agent.get_model_info()), 200

@app.route('/chat/completions', methods=['POST'])
@observe(name="Final answear")
def get_response():
    try:
        data = request.json
        langfuse_cli = get_client()
        sessionID = getSessionID(data)
        langfuse_cli.update_current_trace(session_id=sessionID)

        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        messages = data['messages']
        if not messages:
            return jsonify({"error": "Messages field is required"}), 400
        if not isinstance(messages, list):
            return jsonify({"error": "Messages must be a list"}), 400
        
        stream = data['stream']
        if stream: 
            return Response(agent.opan_ai_stream(agent.stream(messages)), content_type="text/event-stream", headers=setSessionIDHeader({}, sessionID))

        return agent.format_to_openai_response(agent.execute(messages)), 200, setSessionIDHeader({"Content-Type": "application/json"}, sessionID)
    except Exception as e:
        logger.error(e)
        return jsonify({"error": "Internal server error"}), 500

    
if __name__ == '__main__':
    app.run(debug=Env("DEBUG_MODE", False).get(), host="0.0.0.0", port="5000")
