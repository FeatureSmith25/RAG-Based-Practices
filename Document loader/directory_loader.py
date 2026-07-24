from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
loader=DirectoryLoader(
    path=r"C:\Users\hardi_cezwich\OneDrive\Documentos\GitHub\Gen-Ai-Project\RAG Project\Document loader\books",
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

docs=loader.lazy_load()
# print(len(docs))
# print(docs[220].page_content)
# print(docs[220].metadata)

for document in docs:
    print(document.metadata)