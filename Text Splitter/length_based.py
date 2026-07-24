from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader=PyPDFLoader(r"Gen-Ai-Project\RAG Project\Document loader\GRU_Notes.pdf")
text=loader.load()
splitter=CharacterTextSplitter(
    chunk_size=100, 
    chunk_overlap=0,
    separator=''
)
result=splitter.split_documents(text)
print(result)