from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model  = ChatOpenAI(model = 'gpt-4', temperature = 0.5, max_completion_tokens = 20)
# Temp controls the randomness of a language model. Max Completion token restricts the outut of the language Model.

result  = model.invoke("What is the capital of India")
print(result)
print(result.content)