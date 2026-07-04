from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
embeddings=HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en"
)
vector_store=Chroma(
    persist_directory="chroma_db", 
    embedding_function=embeddings
)
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k":4,
        "fetch_k": 10,
        "lambda_mult":0.5
    }
)
llm=ChatMistralAI(model="mistral-small-2603")
prompt=ChatPromptTemplate.from_messages(
    [("system","""You are a helpful AI assistant.
      Use only the provided context to answer the question.
      If the answer is not present in the context,
      say: "I could not find the answer in the document."
      """),
     ("human",
      """Context:{context}
      Question:{question}""")]
)
print("RAg system created ")
print("Press 0 to exit ")
while True:
    query=input("You: ")
    if query =="0":
        break
    docs=retriever.invoke(query)

    context = "".join([docs.page_content for doc in docs]
                      )
    final_prompt=prompt.invoke({
        "context": context,
        "question":query
    })
    response=llm.invoke(final_prompt)
    print(f"\n AI: {response.content}")