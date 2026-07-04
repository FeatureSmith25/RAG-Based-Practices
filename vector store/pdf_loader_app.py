import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(
    page_title="PDF Chatbot",
    page_icon="📚",
    layout="wide"
)

st.title("📚 RAG PDF Chatbot")

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)

if uploaded_file:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    with st.spinner("Processing PDF..."):

        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=10000,
            chunk_overlap=500
        )

        chunks = splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en"
        )
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings
        )

        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 4,
                "fetch_k": 10,
                "lambda_mult": 0.5
            }
        )

    st.success("PDF processed successfully!")

    query = st.text_input(
        "Ask a question about the document"
    )

    if query:

        docs = retriever.invoke(query)

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are a helpful AI assistant.
                    Use only the provided context to answer the question.
                    If the answer is not present in the context,
                    say: "I could not find the answer in the document."
                    """
                ),
                (
                    "human",
                    """
                    Context:
                    {context}

                    Question:
                    {question}
                    """
                )
            ]
        )

        llm = ChatMistralAI(
            model="mistral-small-2603"
        )

        final_prompt = prompt.invoke(
            {
                "context": context,
                "question": query
            }
        )

        response = llm.invoke(final_prompt)

        st.subheader("Answer")
        st.write(response.content)

    os.unlink(pdf_path)
