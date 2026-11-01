"""Web chat page for the real-time agent. Run with:  streamlit run app.py"""

import uuid

import streamlit as st

from agent import Usage, build_agent, run_turn

TOOL_ICONS = {"get_current_time": "🕒", "get_weather": "🌦️", "search_wikipedia": "📚"}

st.set_page_config(page_title="Real-time Agent", page_icon="🤖")


@st.cache_resource
def get_agent():
    # One agent for the app; each browser tab keeps its own thread_id.
    return build_agent()


def new_chat():
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.history = []
    st.session_state.session = Usage()


def usage_caption(u):
    return (
        f"🪙 {u.input:,} in / {u.output:,} out tokens · 🔁 {u.model_calls} model calls · "
        f"🛠️ {u.tool_calls} tool calls · ⏱️ {u.seconds:.1f}s · 💰 ~\\${u.cost:.4f}"
    )


def tool_line(call):
    args = ", ".join(f"{k}={v!r}" for k, v in call["args"].items())
    return f"{TOOL_ICONS.get(call['name'], '🛠️')} `{call['name']}({args})`"


if "thread_id" not in st.session_state:
    new_chat()
agent = get_agent()
session = st.session_state.session

# Sidebar: session usage totals.
with st.sidebar:
    st.header("📊 Session usage")
    c1, c2 = st.columns(2)
    c1.metric("Input tokens", f"{session.input:,}")
    c2.metric("Output tokens", f"{session.output:,}")
    c1.metric("Tool calls", session.tool_calls)
    c2.metric("Est. cost", f"${session.cost:.4f}")
    st.caption("Claude Opus 5.5 · \\$4 in / \\$20 out per million tokens")  # \$ stops Streamlit treating it as LaTeX
    if st.button("🧹 New chat", use_container_width=True):
        new_chat()
        st.rerun()

st.title("🤖 Real-time Agent")
st.caption("LangChain + Claude · live time, weather and Wikipedia")

# Replay the conversation so far.
for item in st.session_state.history:
    with st.chat_message(item["role"]):
        for call in item.get("tools", []):
            st.markdown(tool_line(call))
        st.markdown(item["content"])
        if "usage" in item:
            st.caption(usage_caption(item["usage"]))

if not st.session_state.history:
    st.info("Try: *What's the weather in Leicester right now, and what time is it there?*")

if question := st.chat_input("Ask anything…"):
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        turn, tools = Usage(), []
        tool_area = st.container()

        def text_stream():
            for kind, value in run_turn(agent, question, st.session_state.thread_id, turn):
                if kind == "tool":
                    tools.append(value)
                    tool_area.markdown(tool_line(value))
                else:
                    yield value

        answer = st.write_stream(text_stream())
        st.caption(usage_caption(turn))

    session.add(turn)
    st.session_state.history.append(
        {"role": "assistant", "content": answer, "tools": tools, "usage": turn}
    )
    st.rerun()  # refresh the sidebar totals
