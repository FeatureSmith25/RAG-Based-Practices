from dotenv import load_dotenv
load_dotenv()
from langchain_core.runnables import RunnableSequence
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt1=PromptTemplate(
    template='write a joke about {topic}',
    input_variables=['topic']
)
model=ChatMistralAI()
parser=StrOutputParser()
prompt2=PromptTemplate(
    template='Explain the following joke - {text}',
    input_variables=['text']
)
chain=RunnableSequence(prompt1,model,parser, prompt2, model, parser)

print(chain.invoke({'topic':'AI'}))