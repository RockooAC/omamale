# LLM Agentic System

## Overview

This project implements a multi-agent system architecture for specialized LLM-based agents. The system orchestrates multiple specialized agents to handle complex tasks by coordinating their interactions through a central orchestrator.

## Architecture

The system consists of the following components:

- **Multiagent Orchestrator**: Central coordinator that routes queries to specialized agents
- **Oracle Agent**: Determines which specialized agents should handle specific queries
- **Specialized Agents**:
  - **OTT Specialist**: Expert in multimedia streaming, networking, and system software
  - **Software Developer**: Specializes in programming and technical implementations
  - **Final Answer**: Synthesizes responses from other agents into coherent answers

## Deployment

The system uses Docker Compose for containerized deployment:

```bash
docker-compose up
```

### Services & Ports

- Multiagent: Port 5000 - Main API endpoint
- Oracle: Port 5001
- OTT Specialist: Port 5002
- Software Developer: Port 5003
- Final Answer: Port 5004

## API Usage

Send queries to the main endpoint:

```bash
curl -X POST http://localhost:5000/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is HDR?"}
    ]
  }'
```

## LLM Models

The system leverages different models for different agent roles:

- Oracle: Qwen2.5 7B (Ollama)
- OTT Specialist: Custom model (pipelineRedgeAssistant)
- Software Developer: DeepSeek Coder v2 (Ollama)
- Final Answer: Llama 3.2 1B (Ollama)

## Project Structure

- `/multiagent/`: Orchestration layer and API endpoint
- `/oracle/`: Router agent implementation
- `/ott_specialist/`: Domain-specific agent for OTT/streaming technology
- `/software-developer/`: Programming and implementation agent
- `/final-answer/`: Response synthesis agent
- `/utils/`: Shared utilities and base classes