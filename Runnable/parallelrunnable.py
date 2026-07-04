from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableLambda
load_dotenv()

model=ChatMistralAI(model="mistral-small-2603")

parser=StrOutputParser()

short_prompt=ChatPromptTemplate.from_template(
    "Explain {topic} in 1-2 line"
)

detailed_prompt=ChatPromptTemplate.from_template(
    "Explain {topic} in detail"
)

topic="Machine Learning"
chains=RunnableParallel({"short": RunnableLambda(lambda x :x['short'])| short_prompt | model | parser,
                    "detailed": RunnableLambda(lambda x :x['detailed'])|detailed_prompt | model | parser})

result=chains.invoke({
    "short":{"topic" : "machine learing"},
    "detailed":{"topic" : "deep learing"}})
print(result['short'])
print(result['detailed'])