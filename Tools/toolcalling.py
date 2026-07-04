from dotenv import load_dotenv
load_dotenv()
from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from rich import print
from langchain_core.messages import HumanMessage

# 1. Creating a tool
@tool
def get_text_length(text: str)-> int:
    """Return the number of character"""
    return len(text)

tools={
    "get_text_length":get_text_length
}
llm=ChatMistralAI(model="mistral-small-2603")

# 2. Tool binding
llm_with_tool=llm.bind_tools([get_text_length])
message=[]

# 3. Human message
query=HumanMessage("Return the number of character in the given text 'Hello how are you'")
message.append(query)
print(message)

# 4. AI message
result=llm_with_tool.invoke(message)
message.append(result)
print(message)

# 5. Tool Result
if result.tool_calls:
    print(result.tool_calls[0])
    tool_name=result.tool_calls[0]["name"]
    tool_message=tools[tool_name].invoke(result.tool_calls[0])
    message.append(tool_message)

# 6. Printing result
result=llm_with_tool.invoke(message)
print(result.content)