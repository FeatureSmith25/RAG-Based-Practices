from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
data=PyPDFLoader("Gen-Ai-Project\RAG Project\Document loader\GRU_Notes.pdf")
docs=data.load()
splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=10
)
chunk=splitter.split_documents(docs)
print(chunk[0].page_content)