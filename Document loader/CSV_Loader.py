from langchain_community.document_loaders import CSVLoader

loader=CSVLoader(file_path=r'C:\Users\hardi_cezwich\OneDrive\Documentos\GitHub\Gen-Ai-Project\RAG Project\Document loader\heart.csv')
docs=loader.load()

print(len(docs))
print(docs[1])