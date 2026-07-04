from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

docs = [
    Document(page_content="Gradient descent is an optimization algorithm used in machine learning and deep learning."),
    Document(page_content="Gradient descent minimizes the loss function."),
    Document(page_content="Gradient descent is an optimization technique that minimizes the loss by iteratively updating parameters."),
    Document(page_content="Neural networks use gradient descent for training."),
    Document(page_content="Support Vector Machines are supervised learning algorithms used for classification and regression tasks.")
]
embeddings=HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en"
)

vectorstores=Chroma.from_documents(docs, embeddings)

similarity_retriever=vectorstores.as_retriever(
    search_type="similarity",
    search_kwargs={"k":3}
)

print("\n===== Similarity Search Results ====\n ")
similarity_docs=similarity_retriever.invoke("What is gradient descent")
for doc in similarity_docs:
    print(doc.page_content)

mmr_retriever=vectorstores.as_retriever(
    search_type="mmr",
    search_kwargs={"k":3}
)
print("\n==== MMR Results ====\n")
mmr_docs=mmr_retriever.invoke("what is gradient decsent?")
for doc in mmr_docs:
    print(doc.page_content)