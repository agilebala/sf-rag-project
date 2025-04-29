import os
from dotenv import load_dotenv
from utils.chunking import chunk_transcripts
from utils.embedding import get_embedding
from azure.cosmos import CosmosClient, exceptions, PartitionKey
import pandas as pd

# Load .env file
load_dotenv()

# Azure Cosmos DB config
COSMOSDB_ENDPOINT = os.getenv("AZURE_COSMOSDB_ENDPOINT")
COSMOSDB_KEY = os.getenv("AZURE_COSMOSDB_KEY")
DATABASE_NAME = os.getenv("AZURE_COSMOSDB_DATABASE_NAME")
CONTAINER_NAME = os.getenv("AZURE_COSMOSDB_CONTAINER_NAME")

def main():
    # Connect to Cosmos DB
    client = CosmosClient(COSMOSDB_ENDPOINT, COSMOSDB_KEY)
    database = client.create_database_if_not_exists(id=DATABASE_NAME)

    try:
        container = database.create_container(
            id=CONTAINER_NAME,
            partition_key=PartitionKey(path="/source"),
            offer_throughput=400
        )
        print(f"✅ Container '{CONTAINER_NAME}' created.")
    except exceptions.CosmosResourceExistsError:
        container = database.get_container_client(CONTAINER_NAME)
        print(f"ℹ️ Container '{CONTAINER_NAME}' already exists.")

    # Load transcript chunks
    data_dir = os.path.join(os.getcwd(), "data")
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"❌ Folder not found: {data_dir}")

    print(f"📄 Reading transcripts from: {data_dir}")
    docs = chunk_transcripts(data_dir)

    # Ingest into Cosmos DB
    for idx, doc in enumerate(docs):
        embedding = get_embedding(doc['text'])
        item = {
            "id": f"{doc['source']}_{idx}",
            "text": doc['text'],
            "source": doc['source'],
            "embedding": embedding
        }
        container.upsert_item(item)

    print(f"✅ Indexed {len(docs)} transcript chunks into Cosmos DB.")

if __name__ == "__main__":
    main()



