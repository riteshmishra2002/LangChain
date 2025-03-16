# Q. There are 5 sentences (Documents) and then user asks a query. We need to tell, with which document it matches the most. 
# Ans. Create embedding vectors for those 5 documents, and the create embedding vector for that query. calculate the cosine similarity of query with all the 5 vectors.

#Openai Embedding Models

from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = OpenAIEmbeddings(model = 'text-embedding-3-large', dimensions=300)

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