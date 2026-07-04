from dotenv import load_dotenv
load_dotenv()

import os
import requests

from rich import print
from tavily import TavilyClient

from langchain.tools import tool
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, ToolMessage


# ==========================================================
#                    WEATHER TOOL
# ==========================================================

@tool
def get_weather(city: "str") -> str:
    """Get current weather of a city"""

    API_KEY = os.getenv("OPENWEATHER_API_KEY")

    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?q={city},IN&appid={API_KEY}&units=metric"
    )

    response = requests.get(url)
    data = response.json()

    print("DEBUG:", data)

    if str(data.get("cod")) != "200":
        return f"Error! {data.get('message', 'Could not fetch weather')}"

    temp = data["main"]["temp"]
    desc = data["weather"][0]["description"]

    return f"Weather in {city}: {desc}, {temp}"


print(get_weather.invoke("kanpur"))


# ==========================================================
#                      NEWS TOOL
# ==========================================================

tavily_client = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)


@tool
def get_news(city: "str") -> str:
    """Get latest news about the city"""

    response = tavily_client.search(
        query=f"latest news in {city}",
        search_depth="basic",
        max_results=3,
    )

    result = response.get("results", [])

    print(result)

    if not result:
        return f"No news found for {city}"

    news_list = []

    for r in result:
        title = r.get("title", "No title")
        url = r.get("url", "")
        snippet = r.get("content", "")

        news_list.append(
            f"- {title}\n"
            f" & {url}\n"
            f"  {snippet[:100]}..."
        )

    return f"Latest news in {city}:\n\n" + "\n\n".join(news_list)


print(get_news.invoke("Kanpur"))


# ==========================================================
#                  CREATE LLM & BIND TOOLS
# ==========================================================

llm = ChatMistralAI(
    model="mistral-small-2603"
)

tools_dict = {
    "get_weather": get_weather,
    "get_news": get_news,
}

llm_with_tool = llm.bind_tools(
    [get_weather, get_news]
)


# ==========================================================
#                      AGENT LOOP
# ==========================================================

messages = []

print("[bold cyan]City Intelligence System[/bold cyan]")
print("[yellow]Type Exit to quit[/yellow]")

while True:

    user_input = input("You : ")

    if user_input.lower() == exit:
        break

    messages.append(
        HumanMessage(content=user_input)
    )

    while True:

        result = llm_with_tool.invoke(messages)

        messages.append(result)

        # --------------------------------------------------
        # If Tool Call Required
        # --------------------------------------------------

        if result.tool_calls:

            for tool_call in result.tool_calls:

                tool_name = tool_call["name"]

                # Human In The Loop

                confirm = input(
                    "Agent wants to call {tool_name} Approve(yes/no): "
                )

                if confirm.lower() == "no":
                    print(
                        "Tool call denied and I can't get the latest information."
                    )

                # Execute Tool

                tool_result = tools_dict[tool_name].invoke(tool_call)

                messages.append(
                    ToolMessage(
                        content=tool_result,
                        tool_call_id=tool_call["id"],
                    )
                )

            continue

        # --------------------------------------------------
        # Final Answer
        # --------------------------------------------------

        else:

            print("\n")
            print("=" * 60)
            print("[bold green]Final Answer[/bold green]")
            print("=" * 60)

            print(result.content)

            print("=" * 60)
            print()

            break