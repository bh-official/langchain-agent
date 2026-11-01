"""Real-time agent: a LangChain + Claude chat agent that streams its answer
token by token and calls live-data tools (time, weather, Wikipedia)."""

import time
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessageChunk
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

# Wikipedia rejects requests whose User-Agent has no contact details.
HTTP = httpx.Client(
    timeout=10,
    follow_redirects=True,
    headers={"User-Agent": "langchain-realtime-agent/1.0 (https://github.com/bh-official)"},
)


@tool
def get_current_time(timezone: str = "UTC") -> str:
    """Get the current date and time in an IANA timezone, e.g. 'Europe/London' or 'Asia/Kolkata'."""
    try:
        now = datetime.now(ZoneInfo(timezone))
    except ZoneInfoNotFoundError:
        return f"Unknown timezone '{timezone}'. Use an IANA name like 'Europe/London'."
    return now.strftime("%A %d %B %Y, %H:%M:%S %Z")


@tool
def get_weather(city: str) -> str:
    """Get the live weather for a city (temperature, wind, conditions)."""
    geo = HTTP.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1},
    ).json()
    if not geo.get("results"):
        return f"Could not find a city called '{city}'."
    place = geo["results"][0]
    wx = HTTP.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "current": "temperature_2m,apparent_temperature,relative_humidity_2m,wind_speed_10m,weather_code",
        },
    ).json()["current"]
    return (
        f"{place['name']}, {place.get('country', '')} at {wx['time']} (local): "
        f"{wx['temperature_2m']}°C (feels like {wx['apparent_temperature']}°C), "
        f"humidity {wx['relative_humidity_2m']}%, wind {wx['wind_speed_10m']} km/h, "
        f"WMO weather code {wx['weather_code']}."
    )


@tool
def search_wikipedia(topic: str) -> str:
    """Get a short, up-to-date summary of a topic from Wikipedia."""
    try:
        hits = HTTP.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "opensearch", "search": topic, "limit": 1, "format": "json"},
        ).raise_for_status().json()
        if not hits[1]:
            return f"No Wikipedia article found for '{topic}'."
        title = hits[1][0]
        page = HTTP.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{title.replace(' ', '_')}"
        ).raise_for_status().json()
    except httpx.HTTPError as e:
        return f"Wikipedia lookup failed: {e}"
    return f"{page.get('title', title)}: {page.get('extract', 'No summary available.')}"


SYSTEM_PROMPT = (
    "You are a helpful real-time assistant. Use your tools whenever a question "
    "depends on live data (current time, weather, recent facts) instead of guessing. "
    "Keep answers short and friendly."
)


def build_agent():
    model = ChatAnthropic(model="claude-opus-5-5", max_tokens=16000)
    return create_agent(
        model,
        tools=[get_current_time, get_weather, search_wikipedia],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=InMemorySaver(),  # remembers the conversation per thread
    )


# Claude Opus 5.5 list prices, USD per million tokens.
PRICE_IN, PRICE_OUT = 4.00, 20.00


class Usage:
    """Token / tool / time counters for one turn and for the whole session."""

    def __init__(self):
        self.input = self.output = self.model_calls = self.tool_calls = 0
        self.seconds = 0.0

    def add(self, other):
        for k in ("input", "output", "model_calls", "tool_calls", "seconds"):
            setattr(self, k, getattr(self, k) + getattr(other, k))

    @property
    def cost(self):
        return (self.input * PRICE_IN + self.output * PRICE_OUT) / 1_000_000

    def line(self, label):
        return (
            f"  [{label}] {self.input:,} in / {self.output:,} out tokens | "
            f"{self.model_calls} model calls | {self.tool_calls} tool calls | "
            f"{self.seconds:.1f}s | ~${self.cost:.4f}"
        )


def chat():
    agent = build_agent()
    config = {"configurable": {"thread_id": "cli-session"}}
    session = Usage()
    print("Real-time agent ready. Ask anything (type 'exit' to quit).\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        print("Agent: ", end="", flush=True)
        turn = Usage()
        started = time.perf_counter()
        for mode, data in agent.stream(
            {"messages": [{"role": "user", "content": question}]},
            config,
            stream_mode=["messages", "updates"],
        ):
            if mode == "messages":
                chunk, meta = data
                # Print the model's answer token by token as it is generated.
                if isinstance(chunk, AIMessageChunk) and meta.get("langgraph_node") == "model":
                    print(chunk.text, end="", flush=True)
            elif mode == "updates" and "model" in data:
                # Show each tool call the agent decides to make.
                for msg in data["model"]["messages"]:
                    turn.model_calls += 1
                    if usage := getattr(msg, "usage_metadata", None):
                        turn.input += usage.get("input_tokens", 0)
                        turn.output += usage.get("output_tokens", 0)
                    for call in getattr(msg, "tool_calls", []) or []:
                        turn.tool_calls += 1
                        print(f"\n  [tool] {call['name']}({call['args']})", flush=True)
        turn.seconds = time.perf_counter() - started
        session.add(turn)
        print("\n" + turn.line("this turn"))
        print(session.line("session  ") + "\n")


if __name__ == "__main__":
    chat()
