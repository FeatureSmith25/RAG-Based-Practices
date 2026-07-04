from dotenv import load_dotenv
load_dotenv()

import os
import requests

from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from tavily import TavilyClient


@tool
def get_weather(city:str) -> str:
    """Get Current weather of a city"""
    API_KEY = os.getenv("OPENWEATHER_API_KEY")
    url = f"https://api.openweathermap.org/data/2.5/weather"

    response = requests.get(
        url,
        params={
            "q": city,
            "appid": API_KEY,
            "units": "metric"
        }
    )

    data = response.json()

    # print("DEBUG:", data)

    if str(data.get('cod')) != "200":
        return f"Error: {data.get('message','could not fetch weather')}"
    
    temp = data["main"]['temp']
    desc = data["weather"][0]["description"]
    return f"weather in {city}: {desc}, {temp}°C"

print(get_weather.invoke("Bhopal"))