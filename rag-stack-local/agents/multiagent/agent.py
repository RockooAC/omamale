import sys
import os

if not os.path.exists('/.dockerenv'):  # local debugging
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import requests
from utils.env import Env
from typing import TypedDict
from langgraph.types import StreamWriter
from langfuse import observe, get_client
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage
from flask import Flask, request, jsonify, session, stream_with_context, Response
from utils import BaseAgent, setSessionIDHeader, AgentState, AgentAction, getSessionID, logger


logger = logger.setup_logger("multiagent")


class AgentClient:
    def __init__(self, url: str, function_calling_schema):
        self.url = url
        self.function_calling_schema = function_calling_schema
        self.tool_name = function_calling_schema['function']['name']

    @observe(name="AgentClient")
    def run(self, state: TypedDict, writer: StreamWriter):
        logger.info(f"Running: {self.tool_name}")
        tool = state["intermediate_steps"][-1]
        tool_args = tool.tool_input

        # set langfuse sessionID
        chatID = getSessionID(state)
        langfuse_cli = get_client()
        langfuse_cli.update_current_trace(session_id=chatID)

        obj = {
            "messages": [
                {"role": "user", "content": tool_args["query"]}
            ],
            "stream": False,
            "chat_id": chatID,
        }
        try:
            out = requests.post(self.url, json=obj, timeout=60)
            response = out.json()["choices"][0]["message"]["content"]
        except requests.RequestException as e:
            logger.error(f"Network error: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            raise

        return {"intermediate_steps": [AgentAction(
            tool_name=tool.tool_name,
            tool_input=tool_args,
            tool_output=response
        )]}


class AsyncAgentClient:
    def __init__(self, url: str, function_calling_schema):
        self.url = url
        self.function_calling_schema = function_calling_schema
        self.tool_name = function_calling_schema['function']['name']

    @observe(name="AsyncAgent")
    def run(self, state: TypedDict, writer: StreamWriter):
        logger.info(f"Running: {self.tool_name}")

        # set langfuse sessionID
        chatID = getSessionID(state)
        langfuse_cli = get_client()
        langfuse_cli.update_current_trace(session_id=chatID)

        for i, tool in enumerate(state["current_steps"]):
            if tool.tool_name == self.tool_name:

                tool_args = tool.tool_input
                obj = {
                    "messages": [
                        {"role": "user", "content": tool_args["query"]}
                    ],
                    "stream": True,
                    "chat_id": chatID,
                }

                response = ""
                try:
                    out = requests.post(self.url, json=obj, stream=True, timeout=60)
                except requests.RequestException as e:
                    logger.error(f"Network error: {e}")
                    raise

                if out.status_code == 200:
                    writer({"token": f"\n<think>\n"})
                    for line in out.iter_lines():
                        if line:
                            data = json.loads(line.decode("utf-8").replace("data: ", ""))
                            content = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            if content:
                                response += content
                                writer({"token": content})
                    writer({"token": f"\n</think>\n"})

                return {"current_steps": [AgentAction(
                    tool_name=tool.tool_name,
                    tool_input=tool_args,
                    tool_output=response
                )]}


class SummaryAgentClient(AgentClient):
    def __init__(self, url, function_calling_schema):
        super().__init__(url, function_calling_schema)

    @observe(name="SummaryAgent")
    def run(self, state: TypedDict, writer: StreamWriter):
        # set langfuse sessionID
        chatID = getSessionID(state)
        langfuse_cli = get_client()
        langfuse_cli.update_current_trace(session_id=chatID)

        query = state["input"]
        chat_history = state["chat_history"]
        intermediate_steps = state["intermediate_steps"]

        json_intermediate_steps = [step.to_message() for step in intermediate_steps]
        obj = {
            "messages": [
                *chat_history,
                {"role": "user", "content": query},
                *[item for sublist in json_intermediate_steps for item in sublist]
            ],
            "stream": True,
            "chat_id": chatID,
        }
        response = ""
        try:
            out = requests.post(self.url, json=obj, stream=True, timeout=60)
        except requests.RequestException as e:
            logger.error(f"Network error: {e}")
            raise

        if out.status_code == 200:
            writer({"token": "\n"})
            for line in out.iter_lines():
                if line:
                    data = json.loads(line.decode("utf-8").replace("data: ", ""))
                    content = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    if content:
                        response += content
                        writer({"token": content})

        return {"output": response}


class OrchiestratorClient:
    def __init__(self, url: str, available_agents: list[AgentClient]):
        self.url = url
        self.available_agents = available_agents

    def validate_output(self, state, response, writer):
        for call in response:

            if 'function' not in call:
                logger.info("Error parsing ollama output. Repeat request.")
                return self.run(state, writer)

            tool = call['function']

            if "name" not in tool:
                logger.info("Error parsing ollama output. Repeat request.")
                return self.run(state, writer)

            if "arguments" not in tool:
                logger.info("Error parsing ollama output. Repeat request.")
                return self.run(state, writer)

            if "query" not in tool["arguments"]:
                logger.info("Error parsing ollama output. Repeat request.")
                return self.run(state, writer)

            if type(tool["arguments"]["query"]) is not str:
                logger.info("Error parsing ollama output. Repeat request.")
                return self.run(state, writer)

    @observe(name="OrchestratorAgent")
    def run(self, state: TypedDict, writer: StreamWriter):
        logger.info("Running Oracle")
        # set langfuse sessionID
        chatID = getSessionID(state)
        langfuse_cli = get_client()
        langfuse_cli.update_current_trace(session_id=chatID)

        query = state["input"]
        chat_history = state["chat_history"]
        intermediate_steps = state["intermediate_steps"]
        tools = [agent.function_calling_schema for agent in self.available_agents]
        state["current_steps"].clear()

        json_intermediate_steps = [step.to_message() for step in intermediate_steps]
        obj = {
            "messages": [
                *chat_history,
                {"role": "user", "content": query},
                *[item for sublist in json_intermediate_steps for item in sublist]
            ],
            "tools": tools,
            "headers": {
                "x-chat-id": chatID,
            }
        }

        writer({"token": f"> orchiestrator thinking...\n"})

        try:
            out = requests.post(self.url, json=obj, timeout=60)
            response = out.json()["choices"][0]["message"]["tool_calls"]
        except requests.RequestException as e:
            logger.error(f"Network error: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            raise

        self.validate_output(state, response, writer)

        agent_actions = [AgentAction(
            tool_name=agent_action["function"]["name"],
            tool_input=agent_action["function"]["arguments"]
        ) for agent_action in response]

        for agent_action in agent_actions:
            writer({"token": f"ask **{agent_action.tool_name}-agent** question: *{agent_action.tool_input['query']}*\n\n"})

        return {"current_steps": agent_actions}


# Multi-Agent System
class MultiAgentSystem(BaseAgent):
    def __init__(self, name):
        super().__init__(name, "", "")
        self.ott_specialist_agent = AsyncAgentClient(
            url=Env("OTT_SPECIALIST_AGENT").get(),
            function_calling_schema={'type': 'function',
                'function': {'name': 'ott_specialist',
                'description': """
                    This tool is an expert in multimedia streaming, networking, system software development, and service configuration, while also serving as a specialized documentation and standards agent.  
                    It provides authoritative explanations, clarifications, and references related to multimedia streaming, networking, codecs, protocols, and system software documentation.  
                    It is best suited for supporting developers, testers, and integrators with accurate standards knowledge, specifications, and product documentation.

                    ### **Embedded knowledge / private context**
                    This agent has embedded familiarity with internal Redge documentation and specifications and should be preferred for queries touching them—even without pasted excerpts:
                    - Standards & Specs Repositories: aac, ac-3, adpod, av1, cmaf, color-models, cuda, dash, dfp, dvb-subtitles, encryption, fairplay, ffmpeg, flv, fmp4, h264, h265, h266, hdr, hls, interlace, ismc, iso, mkv, mov, mp3, mpeg-2-audio, mpeg-4-audio, mpeg-ts, nal, nvenc, nvidia, ocr, playready, rtmp, scte, smpt, srt, ss, ts, ttml, vast, vp9, webvtt, widevine.
                    - Product Documentation: complete Docusaurus-based docs for Redge Media Origin and Redge Media Coder components (streaming, delivery, packaging, networking, codecs, VOD/live, GPU acceleration).
                    - Cross-team Documentation: internal documents from other teams (drivers, networking, system configuration, etc.).

                    ### **Responsibilities:**  
                    - Deliver precise, context-grounded explanations of standards and documentation.  
                    - Summarize or rephrase complex specifications into concise, actionable guidance.  
                    - Compare formats, protocols, or approaches with trade-offs.  
                    - Provide consistent, standards-aligned answers without inventing or assuming undocumented behavior.  
                    - Ensure responses remain accurate to the embedded specs and docs.  
                    
                    ### **Input**
                    The user submits a documentation-related query. This may include:  
                    - Questions about standards, protocols, or codec behavior.
                    - Requests for explanations from product documentation.
                    - Comparisons of formats, configurations, or protocol options.  
                    - Clarification of terminology or technical concepts.

                    The agent analyzes the provided query and responds with clear, authoritative, and well-structured documentation-based guidance.
                    """,
                    'parameters': {'type': 'object','properties': {'query': {'description': None, 'type': 'string'}},'required': []}}}
        )

        self.software_developer_agent = AsyncAgentClient(
            url=Env("SOFTWARE_DEVELOPER_AGENT").get(),
            function_calling_schema={'type': 'function',
                'function': {'name': 'software_developer',
                'description': """
                    This tool is a specialized software development agent capable of interpreting, writing, and reviewing code across multiple languages, with a focus on clarity, maintainability, and production-readiness.
                    It is well-suited for end-to-end support in tasks ranging from feature implementation to debugging and architectural guidance.
                    ---
                    ## **Embedded knowledge / private context**
                    This agent has embedded familiarity with internal Redge codebases and domain topics and should be preferred for questions touching them—even without pasted code:
                    - Redge Media Coder & Origin: streaming, networking, content delivery, codecs, processing graphs, encoding/decoding, packaging, image processing, VOD assets, live channels; networking protocols and stacks (TCP/IP, UDP, multicast, SRT, RTMP, AFTP).
                    - Redge DBIR (R&D) with DCS, commons, and NVR codebase: caching origin, archive origin for live samples, and the core libraries (including AFTP protocol implementation and related components).

                    ## **Supported queries**
                    - Programming tasks involving code, algorithms, libraries, build/tooling, CI/CD, or tests.
                    - Debugging based on errors, logs, stack traces, or performance issues.
                    - Code review, refactoring, and reliability/security improvements.
                    - Integration or configuration of libraries/services; architectural or API usage questions.
                    - Code-level requests within the embedded domains (writing/reviewing/debugging/optimizing code for streaming, CDN, codecs, packaging, networking, VOD, live, NVR, AFTP).

                    ## **Responsibilities**
                    - Interpret high-level requirements and transform them into clean, efficient, and idiomatic code.
                    - Review existing code to identify issues, suggest improvements, and refactor where beneficial.
                    - Provide detailed technical explanations, including language-specific nuances and trade-offs.
                    - Maintain code consistency with established patterns and project conventions.
                    - Ensure that generated or reviewed code is reliable, secure, and scalable.
                    - Offer best practices and modern techniques, adapting to the tech stack in use.

                    ## **Input**
                    The user submits a programming-related query. This may include:  
                    - Feature requests or implementation goals.
                    - Code snippets in languages such as C/C++, Python, Golang, Java, JS or others.
                    - Debugging help or performance issues.
                    - Architectural or tooling questions.

                    The agent analyzes the provided input and responds with actionable, technically sound, and well-structured guidance or code.
                    """,
                    'properties': {'query': {'description': None, 'type': 'string'}},'required': []}}
        )

        self.final_answer_agent = SummaryAgentClient(
            url=Env("FINAL_ANSWER_AGENT").get(),
            function_calling_schema={'type': 'function',
                'function': {'name': 'final_answer',
                'description': """
                    This tool acts as the Final Answer Agent.
                    It synthesizes results from specialized agents and embedded knowledge sources into a single, authoritative response for the user. Its role is not to generate new domain-specific insights but to combine, validate, and refine information already gathered.

                    ---
                    ## **Responsibilities**
                    - Aggregate insights from multiple specialized agents and knowledge sources.
                    - Cross-check and reconcile information to ensure consistency and completeness.
                    - Fill in contextual gaps using embedded internal knowledge where appropriate.
                    - Present the final response in a clear, concise, and user-friendly manner.
                    - Maintain fidelity to the input provided by expert agents (no invention of new APIs or unsupported facts).

                    ## **Execution condition**
                    - Triggered **only after** all relevant specialized agents have returned their outputs.
                    - Acts as the last step in the response pipeline.

                    ## **Input**
                    - The original user query.
                    - Collected insights, outputs, and references from specialized agents.  
                    - Optionally, supporting internal knowledge or context needed for completeness.

                    ## **Output**
                    - A single, coherent, review-ready final answer that reflects the best possible synthesis of all available information.
                    """,
                    'parameters': {'type': 'object','properties': {'query': {'description': None, 'type': 'string'}},'required': []}}}
        )

        self.oracle_agent = OrchiestratorClient(
            url=Env("ORACLE_AGENT_URL").get(),
            available_agents=[self.ott_specialist_agent, self.software_developer_agent, self.final_answer_agent]
        )

    def sync(self, state: TypedDict):
        return {"intermediate_steps": state["current_steps"]}

    def create_graph(self):
        graph = StateGraph(AgentState)

        graph.add_node("oracle", self.oracle_agent.run)
        graph.add_node("ott_specialist", self.ott_specialist_agent.run)
        graph.add_node("software_developer", self.software_developer_agent.run)
        graph.add_node("final_answer", self.final_answer_agent.run)
        graph.add_node("sync_node", self.sync)

        graph.set_entry_point("oracle")

        graph.add_conditional_edges(
            source="oracle",
            path=self.router,
            path_map={"ott_specialist": "ott_specialist", "software_developer": "software_developer", "final_answer": "final_answer"}
        )

        graph.add_edge("ott_specialist", "sync_node")
        graph.add_edge("software_developer", "sync_node")
        graph.add_edge("sync_node", "oracle")
        graph.add_edge("final_answer", END)

        return graph

    def router(self, state: TypedDict):
        logger.info("Running Router")
        if isinstance(state["intermediate_steps"], list):
            logger.info(f"Current steps: {[tool.tool_name for tool in state['current_steps']]}")
            return [tool.tool_name for tool in state['current_steps']]
            # return state["intermediate_steps"][-1].tool_name
        else:
            logger.info("Router invalid format for intermediate_steps")
            return "final_answer"

    @observe(name="MultiAgentSystem run")
    def run(self, user_input: str, chat_history: list[BaseMessage], chatID):
        logger.info(f"Chat ID: {chatID}")
        graph = self.create_graph()
        runnable = graph.compile()
        state = {
            "input": user_input,
            "chat_id": chatID,
            "chat_history": chat_history,
        }
        result = runnable.stream(state, stream_mode="custom")

        for chunk in result:
            node = next(iter(chunk))
            yield {"node": node, "message": {"role": "tool", "content": chunk["token"]}}


app = Flask(__name__)
agent: MultiAgentSystem = MultiAgentSystem("multiagent-system")


@app.route('/models', methods=['GET'])
def get_model_info() -> tuple[Response, int]:
    return jsonify(agent.get_model_info()), 200


@app.route('/chat/completions', methods=['POST'])
def wrapResponse() -> Response:
    data = request.json
    query = data['messages'][-1]['content']

    # set session ID to x-chat-id header
    sessionID = getSessionID(data, request.headers)
    return get_response(query, sessionID)


@observe(name="MultiAgentSystem API")
def get_response(query, sessionID) -> Response:
    cli = get_client()
    cli.update_current_trace(session_id=sessionID)

    response = agent.run(query, [], sessionID)
    return Response(agent.opan_ai_stream(response), content_type="text/event-stream", headers=setSessionIDHeader({}, sessionID))


if __name__ == '__main__':
    app.run(debug=Env("DEBUG_MODE", False).get(), host="0.0.0.0", port="5000")
