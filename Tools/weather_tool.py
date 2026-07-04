from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
import requests
load_dotenv()

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import os
from tavily import TavilyClient


@tool
def get_weather(city: str) -> str:
    """You are a helpful weather Chatbot"""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    url = "https://api.openweathermap.org/data/2.5/weather"

    params = {
        "q": city,
        "appid": api_key,
        "units": "metric"
    }

    response = requests.get(url, params=params)
    data = response.json()

    if str(data.get("cod")) != "200":
        return data.get("message", "Unable to fetch weather")

    temp = data["main"]["temp"]
    desc = data["weather"][0]["description"]

    return f"Weather in {city}: {desc}, {temp}°C"


llm = ChatMistralAI(model="mistral-small-2603")
llm_with_tool = llm.bind_tools([get_weather])

messages = []

prompt = input("You: ")
query = HumanMessage(content=prompt)
messages.append(query)

result = llm_with_tool.invoke(messages)
messages.append(result)

tools = {
    "get_weather": get_weather
}

if result.tool_calls:
    tool_name = result.tool_calls[0]["name"]
    tool_args = result.tool_calls[0]["args"]

    tool_result = tools[tool_name].invoke(tool_args)

    tool_message = ToolMessage(
        content=tool_result,
        tool_call_id=result.tool_calls[0]["id"]
    )

    messages.append(tool_message)

    final_result = llm_with_tool.invoke(messages)
    messages.append(final_result)

print(messages['Toolmessage'].content)