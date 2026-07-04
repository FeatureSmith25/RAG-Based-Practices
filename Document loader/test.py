from langchain_community.document_loaders import TextLoader

data=TextLoader("Gen-Ai-Project/RAG Project/Document loader/notes.txt")
# print(data)
docs=data.load()
# print(docs[0].page_content)
print(len(docs))