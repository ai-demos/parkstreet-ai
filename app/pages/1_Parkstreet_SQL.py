from typing import List, Optional, Dict, Any
import streamlit as st


from agno.agent import Agent
from agno.utils.log import logger
from agno.tools.streamlit.components import check_password
from ai.agent import get_analytics_agent

st.set_page_config(
    page_title="Parkstreet SQL",
    page_icon=":rocket:",
)

st.title("Parkstreet SQL")
st.markdown("##### :orange_heart: built using [Agno](https://github.com/agno-agi/agno)")

with st.expander(":rainbow[:point_down: Example Questions]"):
    st.markdown("- Show me some client ids and their cash balances")
    st.markdown("- Show me PO numbers with Approved approval status")


def add_message(role: str, content: str, tool_calls: Optional[List[Dict[str, Any]]] = None) -> None:
    """Safely add a message to the session state"""
    if "messages" not in st.session_state or not isinstance(st.session_state["messages"], list):
        st.session_state["messages"] = []
    st.session_state["messages"].append({"role": role, "content": content, "tool_calls": tool_calls})


def display_tool_calls(tool_calls_container, tools):
    """Display tool calls in a streamlit container with expandable sections.

    Args:
        tool_calls_container: Streamlit container to display the tool calls
        tools: List of tool call dictionaries containing name, args, content, and metrics
    """
    with tool_calls_container.container():
        for tool_call in tools:
            _tool_name = tool_call.get("tool_name")
            _tool_args = tool_call.get("tool_args")
            _content = tool_call.get("content")
            _metrics = tool_call.get("metrics")

            with st.expander(f"🛠️ {_tool_name.replace('_', ' ').title()}", expanded=False):
                if isinstance(_tool_args, dict) and "query" in _tool_args:
                    st.code(_tool_args["query"], language="sql")

                if _tool_args and _tool_args != {"query": None}:
                    st.markdown("**Arguments:**")
                    st.json(_tool_args)

                if _content:
                    st.markdown("**Results:**")
                    try:
                        st.json(_content)
                    except Exception:
                        st.markdown(_content)

                if _metrics:
                    st.markdown("**Metrics:**")
                    st.json(_metrics)


def main() -> None:
    # Get the Agent
    parkstreet_ai: Agent
    if "parkstreet_ai" not in st.session_state or st.session_state["parkstreet_ai"] is None:
        logger.info("---*--- Creating new Analytics Agent ---*---")
        parkstreet_ai = get_analytics_agent()
        st.session_state["parkstreet_ai"] = parkstreet_ai
    else:
        parkstreet_ai = st.session_state["parkstreet_ai"]

    try:
        st.session_state["parkstreet_ai_session_id"] = parkstreet_ai.load_session()
    except Exception:
        st.warning("Could not create Agent session, is the database running?")
        return

    agent_runs = parkstreet_ai.memory.runs
    if len(agent_runs) > 0:
        logger.debug("Loading run history")
        st.session_state["messages"] = []
        for _run in agent_runs:
            if _run.message is not None:
                add_message(_run.message.role, _run.message.content)
            if _run.response is not None:
                add_message("assistant", _run.response.content, _run.response.tools)
    else:
        logger.debug("No run history found")
        st.session_state["messages"] = []

    # Prompt for user input
    if prompt := st.chat_input():
        st.session_state["messages"].append({"role": "user", "content": prompt})
    if st.sidebar.button("Who are you?"):
        _message = "Who are you and what can you do?"
        st.session_state["messages"].append({"role": "user", "content": _message})

    for message in st.session_state["messages"]:
        if message["role"] in ["user", "assistant"]:
            _content = message["content"]
            if _content is not None:
                with st.chat_message(message["role"]):
                    # Display tool calls if they exist in the message
                    # if "tool_calls" in message and message["tool_calls"]:
                    #     display_tool_calls(st.empty(), message["tool_calls"])
                    st.markdown(_content)

    # If last message is from a user, generate a new response
    last_message = st.session_state["messages"][-1] if st.session_state["messages"] else None
    if last_message and last_message.get("role") == "user":
        question = last_message["content"]
        with st.chat_message("assistant"):
            # Create container for tool calls
            tool_calls_container = st.empty()
            resp_container = st.empty()
            with st.spinner("🤔 Thinking..."):
                response = ""
                try:
                    # Run the agent and stream the response
                    run_response = parkstreet_ai.run(question, stream=True)
                    for _resp_chunk in run_response:
                        # Display tool calls if available
                        if _resp_chunk.tools and len(_resp_chunk.tools) > 0:
                            display_tool_calls(tool_calls_container, _resp_chunk.tools)

                        # Display response
                        if _resp_chunk.content is not None:
                            response += _resp_chunk.content
                            resp_container.markdown(response)

                    add_message("assistant", response, parkstreet_ai.run_response.tools)
                except Exception as e:
                    error_message = f"Sorry, I encountered an error: {str(e)}"
                    add_message("assistant", error_message)
                    st.error(error_message)


def restart_agent():
    logger.debug("---*--- Restarting Agent ---*---")
    st.session_state["parkstreet_ai"] = None
    st.session_state["parkstreet_ai_session_id"] = None
    st.rerun()


if check_password():
    main()
