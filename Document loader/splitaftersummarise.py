from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
load_dotenv()
data=PyPDFLoader("Gen-Ai-Project\RAG Project\Document loader\GRU_Notes.pdf")
docs=data.load()
template=ChatPromptTemplate.from_messages(
    [
        ("system","You are a AI that summarizes the text"),
        ("human","{data}")
    ]
)
model=ChatMistralAI(model="mistral-small-2603")
prompt=template.format_messages(data=docs)
result=model.invoke(prompt)
splitter=CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunk=splitter.split_text(result.content)
# print(len(chunk))
print(chunk[0])