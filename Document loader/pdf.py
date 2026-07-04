from langchain_community.document_loaders import PyPDFLoader

data=PyPDFLoader("Gen-Ai-Project\RAG Project\Document loader\GRU_Notes.pdf")
# print(data)
docs=data.load()
print(docs[1].page_content)
# print(len(docs))