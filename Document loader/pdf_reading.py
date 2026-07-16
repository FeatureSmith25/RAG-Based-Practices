from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStore
from langchain_mistralai import ChatMistralAI

document=PyPDFLoader(r"C:\Users\hardi_cezwich\OneDrive\Documentos\GitHub\Gen-Ai-Project\RAG Project\Document loader\SE_File.pdf")
documents=document.load()
embedding=HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en"
)

text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs=text_splitter.split_documents(documents)

vector_store=Chroma.from_documents(documents=docs, embedding=embedding)
# print(vector_store)

query="What are the key take aways from the documents??"

retriever=vector_store.as_retriever()

retriever_docs=retriever.invoke(query)
retrieved_text="\n".join([doc.page_content for doc in retriever_docs])

llm=ChatMistralAI(model="mistral-small-2603")
prompt = f"""
You are an AI assistant.

Answer the following question only using the provided context.

Context:
{retrieved_text}

Question:
{query}
"""
answer=llm.invoke(prompt)

print("Answer: ",answer.content)