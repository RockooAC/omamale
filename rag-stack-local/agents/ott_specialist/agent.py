import sys
import os

if not os.path.exists('/.dockerenv'):  # local debugging
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.env import Env
from langfuse import observe, get_client
from flask import Flask, request, jsonify, Response
from utils import getSessionID, setSessionID, setSessionIDHeader, OpenAiBaseAgent, logger


logger = logger.setup_logger("ott-specialist-agent")


class OttSpecialistAgent(OpenAiBaseAgent):
    def __init__(self, name, host, api_key, model):
        super().__init__(name, host, api_key, model)
    
    @observe()
    def execute(self, query: list[dict]):
        messages = (
            *query,
        )
        response = self.chat(
            messages=messages
        )
        return response
    
    @observe()
    def stream(self, query: list[dict]):
        messages = (
            *query,
        )
        completion = self.chat(messages=messages, stream=True)
        for chunk in completion:
            yield chunk
    

app = Flask(__name__)
agent = OttSpecialistAgent("ott-specialist-agent", Env("OPEN_AI_URL").get(), "0p3n-w3bu!", Env("OPEN_AI_MODEL").get())


@app.route('/models', methods=['GET'])
def get_model_info():
    return jsonify(agent.get_model_info()), 200

@app.route('/chat/completions', methods=['POST'])
@observe(name="OTT Specialist")
def get_response():
    data = request.json
    cli = get_client()
    sessionID = getSessionID(data, request.headers)
    cli.update_current_trace(session_id=sessionID)

    query = data['messages']
    stream = data['stream']
    if stream:
        return Response(agent.opan_ai_stream_to_openai_stream(agent.stream(query)), content_type="text/event-stream", headers=setSessionIDHeader({}, sessionID))
    return agent.format_to_openai_response(agent.execute(query)), 200, setSessionIDHeader({"Content-Type": "application/json"}, sessionID)


if __name__ == '__main__':
    app.run(debug=Env("DEBUG_MODE").get(), host="0.0.0.0", port="5000")
