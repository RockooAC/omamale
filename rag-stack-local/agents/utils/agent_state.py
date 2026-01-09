from typing import TypedDict, Annotated, List, Union
from langchain_core.messages import BaseMessage
from langchain_core.agents import AgentAction
import operator


class AgentState(TypedDict):
    input: str
    chat_id: str
    chat_history: list[BaseMessage]
    current_steps: Annotated[list[tuple[AgentAction, str]], operator.add]
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]
    output: dict[str, Union[str, List[str]]]