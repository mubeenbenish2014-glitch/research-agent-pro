from typing import Literal
from langchain_core.messages import AIMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from hitl import hitl_node
from state import AgentState
from tools import get_tools


def make_agent_node(llm_with_tools):
    def agent_node(state: AgentState) -> dict:
        messages = state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}
    return agent_node


def make_tool_exec_node(tools: list):
    base_tool_node = ToolNode(tools)

    def tool_exec_node(state: AgentState) -> dict:
        decision = state.get("hitl_decision", "approved")

        if decision == "rejected":
            return {}

        if decision == "edited" and state.get("edited_tool_input"):
            new_input = state["edited_tool_input"]
            messages = list(state["messages"])
            for i in range(len(messages) - 1, -1, -1):
                msg = messages[i]
                if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                    old_tc = msg.tool_calls[0]
                    old_args = old_tc.get("args", {})
                    if "query" in old_args:
                        new_args = {**old_args, "query": new_input}
                    else:
                        new_args = {"query": new_input}
                    patched_tc = {**old_tc, "args": new_args}
                    patched_msg = AIMessage(
                        content=msg.content,
                        tool_calls=[patched_tc],
                    )
                    messages[i] = patched_msg
                    break
            patched_state = {**state, "messages": messages}
            return base_tool_node.invoke(patched_state)

        return base_tool_node.invoke(state)

    return tool_exec_node


def should_use_tools(state: AgentState) -> Literal["hitl_gate", "end"]:
    last_msg = state["messages"][-1]
    if isinstance(last_msg, AIMessage) and getattr(last_msg, "tool_calls", None):
        return "hitl_gate"
    return "end"


def after_hitl(state: AgentState) -> Literal["tool_exec", "agent"]:
    decision = state.get("hitl_decision", "approved")
    if decision == "rejected":
        return "agent"
    return "tool_exec"


def after_tool(state: AgentState) -> Literal["agent"]:
    return "agent"


def build_graph(db_path: str = "checkpoints.db"):
    tools = get_tools()
    llm = ChatOllama(model="minimax-m2.5:cloud", temperature=0)
    llm_with_tools = llm.bind_tools(tools)
    builder = StateGraph(AgentState)
    builder.add_node("agent", make_agent_node(llm_with_tools))
    builder.add_node("hitl_gate", hitl_node)
    builder.add_node("tool_exec", make_tool_exec_node(tools))
    builder.set_entry_point("agent")
    builder.add_conditional_edges(
        "agent",
        should_use_tools,
        {"hitl_gate": "hitl_gate", "end": END},
    )
    builder.add_conditional_edges(
        "hitl_gate",
        after_hitl,
        {"tool_exec": "tool_exec", "agent": "agent"},
    )
    builder.add_conditional_edges(
        "tool_exec",
        after_tool,
        {"agent": "agent"},
    )
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
    return graph