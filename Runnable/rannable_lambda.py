from dotenv import load_dotenv
load_dotenv()
from langchain_core.runnables import RunnableSequence, RunnableParallel
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda,RunnablePassthrough

prompt=PromptTemplate(
    template="Write a joke about{topic}",
    input_variables=['topic']
)

model=ChatMistralAI()

parser=StrOutputParser()

joke_generator=RunnableSequence(prompt, model, parser)
def word_counter(text):
    return len(text.split())
parallel_chain=RunnableParallel({
    'joke':RunnablePassthrough(),
    'word_count':RunnableLambda(word_counter)
    # word_count=RunnableLambda(lambda x: len(x.split()))
})

final_chain=RunnableSequence(joke_generator,parallel_chain)
print(final_chain.invoke({'topic':'AI'}))