import os
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from utils.embedding import get_embedding
from azure.cosmos import CosmosClient
import numpy as np

load_dotenv()

COSMOSDB_ENDPOINT = os.getenv("AZURE_COSMOSDB_ENDPOINT")
COSMOSDB_KEY = os.getenv("AZURE_COSMOSDB_KEY")
DATABASE_NAME = os.getenv("AZURE_COSMOSDB_DATABASE_NAME")
CONTAINER_NAME = os.getenv("AZURE_COSMOSDB_CONTAINER_NAME")

client = CosmosClient(COSMOSDB_ENDPOINT, COSMOSDB_KEY)
database = client.get_database_client(DATABASE_NAME)
container = database.get_container_client(CONTAINER_NAME)

def retrieve_top_k(query, k=5):
    query_embedding = get_embedding(query)

    sql = f"""
    SELECT TOP {k} c.text, c.source
    FROM c
    ORDER BY VECTOR_COSINE_DISTANCE(c.embedding, @query_vector)
    """

    params = [{"name": "@query_vector", "value": query_embedding}]

    results = container.query_items(
        query=sql,
        parameters=params,
        enable_cross_partition_query=True
    )

    return [{"text": r["text"], "source": r["source"]} for r in results]
