from dotenv import load_dotenv
load_dotenv()
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# prompt
prompt=ChatPromptTemplate.from_template("Explain {topic} in simple words")

# model
model=ChatMistralAI(model="mistral-small-2603")

# outputparser
parser=StrOutputParser()

# formatted_prompt=prompt.format_messages(topic="Machine Learning")

# response=model.invoke(formatted_prompt)

# result=parser.invoke(response.content)

chain=prompt | model | parser

result=chain.invoke("machine learning")
print(result)