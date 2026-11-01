# Real-time LangChain Agent

A simple agentic chat app built with **LangChain 1.x** and **Claude (Opus 5.5)**.

- **Streams** the answer token by token as it is generated
- **Decides on its own** which tools to call, and shows each call
- **Live data tools:** current time (any timezone), live weather (Open-Meteo), Wikipedia summaries. No extra API keys needed.
- **Remembers** the conversation within a session (LangGraph checkpointer)
- **Usage stats** under every answer: tokens in/out, model calls, tool calls, time, and estimated cost, for the turn and the whole session:

```
  [this turn] 1,200 in / 300 out tokens | 2 model calls | 1 tool calls | 3.4s | ~$0.0108
  [session  ] 3,950 in / 870 out tokens | 5 model calls | 2 tool calls | 9.1s | ~$0.0332
```

## How it works

```
You ──► create_agent (LangChain) ──► Claude decides
                 ▲                     │
                 │      tool result     ▼
                 └──── get_current_time / get_weather / search_wikipedia
```

The agent loops: model → tool call → tool result → model … until it can answer, streaming the final text to the terminal.

## Run it

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env      # then put your Anthropic API key in .env
python agent.py
```

Try:

- `What's the weather in Leicester right now, and what time is it there?`
- `Tell me about Amazon S3, then compare the weather in London and Chennai.`

## Add your own tool

```python
@tool
def my_tool(arg: str) -> str:
    """Describe what it does - Claude reads this to decide when to call it."""
    ...
```

Then add it to the `tools=[...]` list in `build_agent()`.
