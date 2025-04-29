import os
from dotenv import load_dotenv
from openai import OpenAIClient 
from azure.core.credentials import AzureKeyCredential

load_dotenv()

# Get values from .env
api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

# Initialize OpenAI client
client = OpenAIClient(endpoint=endpoint, credential=AzureKeyCredential(api_key))


def get_embedding(text: str):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=[text]
    )
    return response.data[0].embedding


def generate_answer(query: str, context_chunks: list):
    context = "\n\n".join(chunk["text"] for chunk in context_chunks)
    prompt = f"Answer the question based on the context below.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"

    response = client.chat.completions.create(
        model="gpt-35-turbo",  # for Azure OpenAI
        messages=[
            {"role": "system", "content": "You are a helpful assistant for summarizing Salesforce earnings call transcripts."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=500
    )
    return response.choices[0].message.content.strip()
