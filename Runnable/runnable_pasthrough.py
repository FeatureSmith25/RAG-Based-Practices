from langchain_core.prompts import PromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableSequence
from dotenv import load_dotenv
load_dotenv()

model=ChatMistralAI(model="mistral-small-2603")
prompt1=PromptTemplate(
    template='write a joke about {topic}',
    input_variables=['topic']
)
parser=StrOutputParser()
prompt2=PromptTemplate(
    template='Explain the following joke - {text}',
    input_variables=['text']
)

joke_generator=RunnableSequence(prompt1, model, parser)

parallel_chain=RunnableParallel({
    'joke': RunnablePassthrough(),
    'explaination': RunnableSequence(prompt2 ,model, parser)
})

final_chain=RunnableSequence(joke_generator, parallel_chain)
print("Joke: ",final_chain.invoke({'topic':'cricket'})['joke'])
print("Explaination: ",final_chain.invoke({'topic':'cricket'})['explaination'])