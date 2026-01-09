import sys
import os

if not os.path.exists('/.dockerenv'):  # local debugging
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.env import Env
from langfuse import observe, get_client
from flask import Flask, request, jsonify
from utils import OllamaBaseAgent, getSessionID, setSessionIDHeader, logger


logger = logger.setup_logger("oracle-agent")


# Oracle Agent
class OracleAgent(OllamaBaseAgent):
    def __init__(self, name, host, model):
        super().__init__(name, host, model)

        self.system_prompt = """
        You are the Oracle, the decision-making AI responsible for delegating tasks to the most appropriate specialized agent.  
        Your role is to assess the user's query and determine which tool or tools should be used to gather the necessary information to provide an accurate response.  
        
        **Responsibilities:**  
        - Analyze the user's query thoroughly to determine its relevance to OTT television, software development, or if it requires a final response.  
        - Assign the task to the most suitable specialized agent for gathering the necessary information.  
        - Do not answer the user's query directly; instead, delegate tasks to the appropriate agents.  

        **Rules of Engagement:**  
        - After each interaction with a specialized tool, reassess whether additional tools are required or if enough information has been gathered to proceed to the Final Answer.  
        - Limit tool interactions to a maximum of three per tool. If sufficient information is not gathered, move forward to the Final Answer.  
        - Ensure efficient and accurate delegation to streamline the process of obtaining a comprehensive answer.  
        """

    @observe()
    def decide_action(self, income_messages: list[dict], tools: list[dict]):
        messages = [
            {"role": "system", "content": self.system_prompt},
            *income_messages,
        ]
        response = self.chat(messages=messages, tools=tools, stream=False)
        return response

    def validate_response(self, response):
        if response.message.tool_calls is None or len(response.message.tool_calls) < 1:
            return False
        return True


app = Flask(__name__)
agent = OracleAgent("oracle-agent", Env("OLLAMA_URL").get(), Env("OLLAMA_MODEL").get())

@app.route('/models', methods=['GET'])
def get_model_info():
    return jsonify(agent.get_model_info()), 200


@app.route('/chat/completions', methods=['POST'])
@observe(name="Oracle")
def get_response():
    data = request.json
    sessionID = getSessionID(data, request.headers)
    get_client().update_current_trace(session_id=sessionID)

    messages = data['messages']
    tools = data["tools"]
    maximum_function_calling_repeat = 3
    for _ in range(maximum_function_calling_repeat):
        response = agent.decide_action(messages, tools)
        if agent.validate_response(response):
            break
        logger.error("Failed tool calling. Repeat request")

    return agent.format_to_openai_response(response), 200, setSessionIDHeader({"Content-Type": "application/json"}, sessionID)


if __name__ == '__main__':
    app.run(debug=Env("DEBUG_MODE", False).get(), host="0.0.0.0", port="5000")
