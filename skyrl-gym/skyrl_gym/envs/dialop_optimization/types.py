from typing import Literal, TypedDict, Optional

AgentType = Literal["user", "assistant"]

DialOpOptimizationActionType = Literal["chat", "propose_solution", "accept", "reject"]


class HistoryElement(TypedDict):
    agent: AgentType
    action_type: DialOpOptimizationActionType
    action_content: Optional[str]
    message: str  # Raw message from the agent
    error_message: Optional[str]
