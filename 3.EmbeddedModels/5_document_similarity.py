# Q. There are 5 sentences (Documents) and then user asks a query. We need to tell, with which document it matches the most. 
# Ans. Create embedding vectors for those 5 documents, and the create embedding vector for that query. calculate the cosine similarity of query with all the 5 vectors.

#Open source Embedding Models

from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

embedding = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

documents = [

    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = 'Tell me about Virat Kohli'

doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

print(cosine_similarity([query_embedding], doc_embeddings))