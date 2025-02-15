import os
import json
import re
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI
from groq import Groq

# Load environment variables
load_dotenv()

# Initialize API clients
pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))  # Pinecone API Key
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # OpenAI for embeddings
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))  # Groq for reranking

# Define the index name
index_name = "k-electric-dataset"

# Load the dataset and generate embeddings
def initialize_embeddings():
    file_path = "KE.csv"  # Ensure correct path
    df = pd.read_csv(file_path)

    texts = df.apply(lambda row: f"Q: {row['Questions']} A: {row['Answers']}", axis=1).tolist()

    # Generate embeddings using OpenAI
    response = openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=texts,
        encoding_format="float"
    )

    # Extract embeddings
    df['embedding'] = [item.embedding for item in response.data]

    # Check if Pinecone index exists before creating
    existing_indexes = [index["name"] for index in pinecone.list_indexes()]
    if index_name not in existing_indexes:
        pinecone.create_index(
            name=index_name,
            dimension=len(df['embedding'][0]),  # Extract embedding size dynamically
            metric="cosine",
            spec={"serverless": {"cloud": "aws", "region": "us-east-1"}}
        )
        print(f" Created new Pinecone index: {index_name}")
    else:
        print(f"ℹ️ Pinecone index '{index_name}' already exists. Skipping creation.")

    # Connect to the index
    index = pinecone.Index(index_name)

    # Upsert embeddings into Pinecone
    records = [
        {
            "id": str(i),
            "values": row["embedding"],
            "metadata": {
                "question": row["Questions"],
                "answer": row["Answers"],
                "genre": row["Genre"],
                "filename": row["Filename"]
            }
        }
        for i, row in df.iterrows()
    ]

    index.upsert(vectors=records)
    print(" Embeddings stored in Pinecone!")

# Function for searching and reranking
def search_and_rerank(query_text: str, top_k: int = 5):
    # Connect to Pinecone index
    index = pinecone.Index(index_name)

    # Step 1: Convert query into an embedding
    query_embedding = openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=query_text,
        encoding_format="float"
    ).data[0].embedding

    # Step 2: Query Pinecone for similar results
    search_results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_values=False,
        include_metadata=True
    )

    # Step 3: Extract metadata from results
    documents = [
        {
            "id": match["id"],
            "genre": match["metadata"]["genre"],  
            "filename": match["metadata"]["filename"],
            "answer": match["metadata"]["answer"]
        }
        for match in search_results["matches"]
    ]

    # Step 4: Rerank results using **Groq LLM**
    rerank_response = groq_client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a JSON-only responder. Given a query and a list of documents, return the most relevant ones in order as a JSON array."
                    "DO NOT include explanations, additional text, or formatting. Return only a **valid JSON array**."
                    "Each item must contain 'id', 'genre', 'filename', and 'answer'."
                    "Format strictly as follows:\n"
                    '[{"id": "rec1", "genre": "Energy Sector", "filename": "energy-strategy.pdf", "answer": "K-Electric provides electricity to Karachi..."}]'
                )
            },
            {
                "role": "user",
                "content": f"Query: {query_text}\nDocuments: {json.dumps(documents)}\nReturn only a JSON list, no extra text."
            }
        ],
        max_tokens=500
    )

    # Step 5: Extract and clean Groq response
    try:
        response_text = rerank_response.choices[0].message.content.strip()

        # Handle cases where Groq adds ```json wrappers
        if response_text.startswith("```json"):
            response_text = response_text[7:-3].strip()

        # Use regex to extract JSON part if extra text is included
        match = re.search(r'\[\s*\{.*\}\s*\]', response_text, re.DOTALL)
        if match:
            response_text = match.group(0)

        # Load the cleaned JSON response
        reranked_results = json.loads(response_text)

    except json.JSONDecodeError:
        print(f" Error: Failed to parse reranked response.\n🔍 Raw Output:\n{response_text}")
        reranked_results = []

    return reranked_results
