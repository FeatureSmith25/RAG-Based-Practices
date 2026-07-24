from langchain_community.document_loaders import PyPDFLoader
loader=PyPDFLoader(f"Gen-Ai-Project\RAG Project\Document loader\GRU_Notes.pdf")
docs=loader.load()
print(docs[0].page_content)
print(docs[1].metadata)