from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

data=PyPDFLoader(r"C:\Users\hardi_cezwich\OneDrive\Documentos\GitHub\Gen-Ai-Project\RAG Project\vector store\GRU_Notes.pdf")
docs=data.load()
embedding=HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en"
)

spliter=RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=10
)

split_content=spliter.split_documents(docs)
vectorstores = Chroma.from_documents(
    documents=split_content,
    embedding=embedding
)

print("\n==== Similarity Search Result ====\n")
similarity_retriever=vectorstores.as_retriever(
    search_type="similarity",
    search_kwargs={"k":3}
)

input1=input("Enter your questions: ")
similarity_docs=similarity_retriever.invoke(input1)
for doc in similarity_docs:
    print(doc.page_content)

print("\n==== MMR search Result ====\n")
MMR_retriever=vectorstores.as_retriever(
    search_type="mmr",
    search_kwargs={"k":3}
)
input2=input("Enter your questions: ")
MMR_docs=MMR_retriever.invoke(input2)
for doc in MMR_docs:
    print(doc.page_content)