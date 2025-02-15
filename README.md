# K-Electric Search API

A **FastAPI-powered AI API** that allows users to **search, retrieve, and rerank** K-Electric dataset records using **Pinecone vector database, OpenAI embeddings (`text-embedding-ada-002`), and Groq LLM (`mixtral-8x7b-32768`)**.

---

##  Features
✅ Generates **embeddings** from the dataset using OpenAI.  
✅ Stores embeddings in **Pinecone** for efficient search.  
✅ Retrieves top-matching records and **reranks results using Groq AI**.  
✅ Provides a **FastAPI-based REST API** for querying the dataset.  


Create a Virtual Environment

python -m venv venv
venv\Scripts\activate     # On Windows


Install Dependencies
pip install -r requirements.txt

Set Up Environment Variables (.env)
Create a .env file and add your API keys:

PINECONE_API_KEY=your_pinecone_api_key

OPENAI_API_KEY=your_openai_api_key

GROQ_API_KEY=your_groq_api_key

Running the Project
1️⃣ Start the FastAPI Server
bash
Copy
Edit
python FastAPI_server.py
2️⃣ Test the API
Open your browser and go to:

arduino
Copy
Edit
http://127.0.0.1:8000/docs
Use the Swagger UI to send search queries.

3️⃣ API Example Usage (cURL)
bash
Copy
Edit
curl -X 'POST' \
  'http://127.0.0.1:8000/search' \
  -H 'Content-Type: application/json' \
  -d '{"query": "How does K-Electric distribute power?", "top_k": 5}'
4️⃣ Expected JSON Response
json
Copy
Edit
{
    "query": "How does K-Electric distribute power?",
    "results": [
        {
            "id": "15",
            "genre": "Power Distribution",
            "filename": "distribution-overview.pdf",
            "answer": "K-Electric distributes power using an advanced smart grid..."
        }
    ]
}
🛠 API Endpoints
🔹 POST /search
Description: Search the dataset and retrieve the most relevant results.
Request Body:
json
Copy
Edit
{
  "query": "How does K-Electric distribute power?",
  "top_k": 5
}
Response:
json
Copy
Edit
{
  "query": "How does K-Electric distribute power?",
  "results": [
    {"id": "15", "genre": "Power Distribution", "filename": "distribution-overview.pdf", "answer": "K-Electric distributes power..."}
  ]
}
🔹 GET /
Description: API Health Check
Response: { "message": "K-Electric Search API is running!" }
