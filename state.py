# Import Annotated for adding metadata to type hints
from typing import Annotated

# add_messages is a reducer — it appends new messages instead of replacing
from langgraph.graph.message import add_messages

# TypedDict lets us define a dictionary with fixed keys and types
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """
    Shared state that flows through every node in the LangGraph graph.
    Every node can read from and write to this state.
    """

    # Full conversation history (human + AI + tool messages)
    # add_messages reducer ensures messages are appended, not overwritten
    messages: Annotated[list, add_messages]

    # Stores human decision after HITL review: "approved", "edited", or "rejected"
    hitl_decision: str | None

    # Stores the reason if human rejected a tool call
    rejection_reason: str | None

    # Stores the new query if human edited the tool call input
    edited_tool_input: str | None