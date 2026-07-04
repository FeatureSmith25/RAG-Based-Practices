from dotenv import load_dotenv
load_dotenv()
from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from rich import print

def get_text_length(text:str)->int:
    """ Returns the number of character in a given text"""
    return len(text)
llm=ChatMistralAI(model="mistral-small-2603")
#  tool binding
llm_with_tool=llm.bind_tools([get_text_length])
result=llm.invoke("hello")
result1=llm_with_tool.invoke("hello")
print(result)
print()
print()
print()
print(result1)