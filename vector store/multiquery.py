from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from dotenv import load_dotenv
load_dotenv()


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
vectorstore=Chroma.from_documents(docs,embeddings)
retriever=vectorstore.as_retriever()

llm=ChatMistralAI(model="mistral-small-latest")
multi_query_retriever=MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=llm
)
query="What is gradient descent"
docs=multi_query_retriever.invoke(query)
print("\nRetrieved Documents:\n")
for doc in docs:
    print(doc.page_content)