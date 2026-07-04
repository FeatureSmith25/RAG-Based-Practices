from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from dotenv import load_dotenv
load_dotenv()

model=ChatMistralAI(model="mistral-small-2603")
parser=StrOutputParser()

code_prompt=ChatPromptTemplate.from_messages([
    ("system", "You are a code generator"),
    ("human", "{topic}")
])

Explain_prompt=ChatPromptTemplate.from_messages({
    ("system", "You are a helpful assistant who explain code in simple terms"),
    ("human","Explain the following code in simple words:\n{code}")
})

seq=code_prompt | model | parser 

seq1=RunnableParallel(
    {"code":RunnablePassthrough(),
     "explaination": Explain_prompt | model | parser
    }
)
chain=seq | seq1
result=chain.invoke({"topic":"Please write a code of Palindrome in python"})
print(result['code'])
print(result['explaination'])