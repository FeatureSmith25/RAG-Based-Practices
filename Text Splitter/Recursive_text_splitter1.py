from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
text="""from dotenv import load_dotenv
load_dotenv()
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# Prompt Template
prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in simple words"
)

# Model
model = ChatMistralAI(model="mistral-small-2603")

# Output Parser
parser = StrOutputParser()

chain=prompt | model | parser
result=chain.invoke("Machine learning")
print(result)"""
splitter=RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=100, 
    chunk_overlap=0
)
chunk=splitter.split_text(text)
print(len(chunk))
print(chunk)