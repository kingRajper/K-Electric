import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from KE_Embeddings import initialize_embeddings, search_and_rerank

# Initialize embeddings when the server starts
initialize_embeddings()

# Initialize FastAPI
app = FastAPI(title="K-Electric Search API", version="1.0", description="API to search K-Electric dataset and rerank results.")

# Request model for query input
class SearchQuery(BaseModel):
    query: str
    top_k: int = 5

# API endpoint to perform search and reranking
@app.post("/search", summary="Search and rerank results from K-Electric dataset")
async def search_api(request: SearchQuery):
    results = search_and_rerank(request.query, request.top_k)
    return {"query": request.query, "results": results}

# Root endpoint
@app.get("/", summary="API Health Check")
async def root():
    return {"message": "K-Electric Search API is running!"}

# Run FastAPI server
if __name__ == "__main__":
    uvicorn.run("FastAPI_server:app", host="127.0.0.1", port=8000, reload=True)
