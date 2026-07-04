from langchain_community.retrievers import ArxivRetriever
retriever=ArxivRetriever(
    load_max_docs=2,
    load_all_available_meta=True
)
docs = retriever.invoke("More about LLMs")
for i, doc in enumerate(docs):
    print(f"\nResult {i+1}")
    print("Title: ", doc.metadata.get("Title"))
    print("ArxivID: ", doc.metadata.get("ArxivID"))
    print("Summary",doc.page_content[:500])
