from langchain_community.document_loaders import WebBaseLoader
from langchain_mistralai import ChatMistralAI
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = ChatMistralAI(model="mistral-small-2603")
embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")

website = input("Enter your website: ")
url = website

data = WebBaseLoader(url)
docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

split_content = splitter.split_documents(docs)

vector_stores = Chroma.from_documents(
    documents=split_content,
    embedding=embedding
)

print("\n==== Similarity Search Result ====\n")

similarity_retriever = vector_stores.as_retriever(
    search_type="similarity",
    search_kwargs={"k":4}
)

question = input("Enter your question: ")

similarity_docs = similarity_retriever.invoke(question)

context = "\n\n".join([doc.page_content for doc in similarity_docs])

prompt = f"""
Answer the question using only the following context.

Context:
{context}

Question:
{question}
"""

response = llm.invoke(prompt)

parser = StrOutputParser()

final_answer = parser.invoke(response)

print("\n===== Final Answer =====\n")
print(final_answer)