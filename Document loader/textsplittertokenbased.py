from langchain_text_splitters import TokenTextSplitter
from langchain_community.document_loaders import PyPDFLoader
Splitter=TokenTextSplitter(
    chunk_size=100,
    chunk_overlap=10
)
data=PyPDFLoader("Gen-Ai-Project\RAG Project\Document loader\GRU_Notes.pdf")
docs=data.load()
chunk=Splitter.split_documents(docs)
# print(len(chunk))
print(chunk[0].page_content)