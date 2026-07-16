from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel
from dotenv import load_dotenv
load_dotenv()

prompt1=PromptTemplate(
    template='Generate a tweet about {topic}',
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template='Generate a Linkedin post about {topic}',
    input_variables=['topic']
)

model=ChatMistralAI()
parser=StrOutputParser()

parallel_chain=RunnableParallel({
    'tweet': RunnableSequence(prompt1, model, parser), 
    'linkedin': RunnableSequence(prompt2, model, parser)
})
result=parallel_chain.invoke({'topic':'AI'})
print("Tweet: ",result['tweet'])
print("Linkedin: ",result['linkedin'])