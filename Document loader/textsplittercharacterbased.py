from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
splitter=CharacterTextSplitter(
    separator="",
    chunk_size=10,
    chunk_overlap=1
)
data=TextLoader("Gen-Ai-Project/RAG Project/Document loader/notes.txt")
docs=data.load()
chunk=splitter.split_documents(docs)
print(len(chunk))

for i in chunk:
    print(i.page_content)
    print()