from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel
from langchain_community.document_loaders import WebBaseLoader
from dotenv import load_dotenv
load_dotenv()
model=ChatMistralAI()
parser=StrOutputParser()
prompt=PromptTemplate(
    template="Answer the following question - \n {question} from the following text - \n {text}",
    input_variables={'question','text'}
)

url=f"https://www.apple.com/in/macbook-air/?afid=p240%7Cgo~cmp-11182149775~adg-109263622053~ad-780589903439_kwd-5029010249~dev-c~ext-~prd-~mca-~nt-search&cid=aos-in-kwgo-txt-mac-mac--"
loader=WebBaseLoader(url)

docs=loader.load()
print(len(docs))
print(docs[0].page_content)

chain= prompt | model | parser
print(chain.invoke({'question':'What is the product that we are talking about?', 'text': docs[0].page_content}))