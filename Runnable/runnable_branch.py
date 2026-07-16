from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnableBranch, RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_mistralai import ChatMistralAI
load_dotenv()

prompt1=PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template='Summarise the following {text}',
    input_variables=['text']
)
model=ChatMistralAI()
parser=StrOutputParser()
report_gen_chain=RunnableSequence(prompt1,model,parser)

branch_chain=RunnableBranch(
    (lambda x: len(x.split())>500, RunnableSequence(prompt2, model, parser)),
    RunnablePassthrough()
)
finalchain=RunnableSequence(report_gen_chain, branch_chain)
print(finalchain.invoke({'topic':'russia vs ukrain war'}))