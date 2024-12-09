from typing import List
import streamlit as st


from phi.agent import Agent
from phi.utils.log import logger
from ai.agent import get_analytics_agent

st.set_page_config(
    page_title="Parkstreet AI!",
    page_icon=":rocket:",
)

st.title("Parkstreet AI!")
st.markdown("##### :peanuts: built using [phidata](https://github.com/phidatahq/phidata)")

with st.expander(":rainbow[:point_down: Example Questions]"):
    st.markdown("- Show me the shipments departing in the next 30 days?")


def main() -> None:
    # Get the Agent
    parkstreet_ai: Agent
    if "parkstreet_ai" not in st.session_state or st.session_state["parkstreet_ai"] is None:
        if (
            "parkstreet_ai_session_id" in st.session_state
            and st.session_state["parkstreet_ai_session_id"] is not None
        ):
            logger.info("---*--- Reading Analytics Agent ---*---")
            parkstreet_ai = get_analytics_agent(
                team_id="2", run_id=st.session_state["parkstreet_ai_session_id"]
            )
        else:
            logger.info("---*--- Creating new Analytics Agent ---*---")
            parkstreet_ai = get_analytics_agent(team_id="2")
        st.session_state["parkstreet_ai"] = parkstreet_ai
    else:
        parkstreet_ai = st.session_state["parkstreet_ai"]

    # Create Agent run (i.e. log to database) and save run_id in session state
    st.session_state["parkstreet_ai_session_id"] = parkstreet_ai.create_session()

    # Load existing messages
    agent_chat_history = parkstreet_ai.memory.get_messages()
    if len(agent_chat_history) > 0:
        logger.debug("Loading chat history")
        st.session_state["messages"] = agent_chat_history
    else:
        logger.debug("No chat history found")
        st.session_state["messages"] = [{"role": "assistant", "content": "Ask me about procurement data."}]

    # Prompt for user input
    if prompt := st.chat_input():
        st.session_state["messages"].append({"role": "user", "content": prompt})
    if st.sidebar.button("Who are you?"):
        _message = "Who are you and what can you do?"
        st.session_state["messages"].append({"role": "user", "content": _message})

    # Display existing chat messages
    for message in st.session_state["messages"]:
        if message["role"] == "system":
            continue
        if message["role"] == "tool":
            continue
        with st.chat_message(message["role"]):
            content = message.get("content")
            if content:
                st.write(content)

    # If last message is from a user, generate a new response
    last_message = st.session_state["messages"][-1]
    if last_message.get("role") == "user":
        question = last_message["content"]
        with st.chat_message("assistant"):
            with st.spinner("Working..."):
                response = ""
                resp_container = st.empty()
                for delta in parkstreet_ai.run(question, stream=True):
                    response += delta.content  # type: ignore
                    resp_container.markdown(response)
                st.session_state["messages"].append({"role": "assistant", "content": response})

    st.sidebar.markdown("---")

    if st.sidebar.button("New Session"):
        restart_agent()
    if st.sidebar.button("Auto Rename Session"):
        parkstreet_ai.auto_rename_session()
    if parkstreet_ai.storage:
        analytics_assistant_run_ids: List[str] = parkstreet_ai.storage.get_all_session_ids()
        new_analytics_assistant_run_id = st.sidebar.selectbox(
            "Session ID", options=analytics_assistant_run_ids
        )
        if st.session_state["parkstreet_ai_session_id"] != new_analytics_assistant_run_id:
            logger.info(f"Loading run {new_analytics_assistant_run_id}")
            st.session_state["parkstreet_ai"] = get_analytics_agent(
                team_id="2",
                run_id=new_analytics_assistant_run_id,
                debug_mode=True,
            )
            st.rerun()
    analytics_agent_run_name = parkstreet_ai.session_name
    if analytics_agent_run_name:
        st.sidebar.write(f":thread: {analytics_agent_run_name}")
    # if st.sidebar.button("Load knowledge base"):
    #     load_analytics_ai_knowledge_base()


def restart_agent():
    logger.debug("---*--- Restarting Agent ---*---")
    st.session_state["parkstreet_ai"] = None
    st.session_state["parkstreet_ai_session_id"] = None
    st.rerun()


main()
