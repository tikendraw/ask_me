import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from llama_index.core import Settings, StorageContext
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.litellm import LiteLLM
from pydantic import BaseModel

from .base.logger import Logger
from .rag_implementation import WebRAG

load_dotenv('../.env')
logger = Logger().get_logger()


# Initialize LLM and embedding model
try: 
    model = os.getenv("DEFAULT_MODEL")
    api_key =os.getenv('DEFAULT_MODEL_API_KEY')
    if model is None or api_key is None:
        raise Exception("DEFAULT_MODEL or DEFAULT_MODEL_API_KEY environment variable not set. Please set it.")
    else:
        model='HEHE'
except:
    logger.error("DEFAULT_MODEL environment variable not set. Please set it.")
    exit()

llm = LiteLLM(temperature=0, model=model, api_key=api_key)
embed_model = GeminiEmbedding()
Settings.embed_model = embed_model
Settings.llm = llm

app = FastAPI()

# Global dictionary to hold per-user RAGs
user_rags: Dict[str, WebRAG] = {}

# Pydantic models
class IngestRequest(BaseModel):
    urls: List[str]

class QueryRequest(BaseModel):
    query: str

def get_user_rag(user_id: str) -> WebRAG:
    """Get or create a RAG instance for a user"""
    if user_id not in user_rags:
        try:
            storage_context = StorageContext.from_defaults()
            user_rags[user_id] = WebRAG(
                user_id=user_id,
                llm=Settings.llm,
                storage=storage_context,
                embed_model=Settings.embed_model
            )
            logger.info(f"Created new RAG instance for user {user_id}")
        except Exception as e:
            logger.error(f"Failed to create RAG for user {user_id}: {str(e)}")
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to initialize RAG system: {str(e)}"
            )
    return user_rags[user_id]

@app.post("/ingest")
async def ingest_data(
    request: IngestRequest,
    x_user_id: Optional[str] = Header(None, description="Unique identifier for the user")
):
    """Ingest URLs for a specific user"""
    if not x_user_id:
        raise HTTPException(
            status_code=400, 
            detail="User ID header (X-User-ID) is required"
        )
    
    if not request.urls:
        raise HTTPException(
            status_code=400, 
            detail="No URLs provided"
        )
    
    try:
        rag = get_user_rag(x_user_id)
        rag.ingest(request.urls)
        return {
            "status": "success",
            "message": f"Successfully ingested {len(request.urls)} URLs",
            "user_id": x_user_id
        }
    except Exception as e:
        logger.error(f"Ingestion failed for user {x_user_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {str(e)}"
        )

@app.post("/query")
async def query_data(
    request: QueryRequest,
    x_user_id: Optional[str] = Header(None, description="Unique identifier for the user")
):
    """Query the RAG system for a specific user"""
    if not x_user_id:
        raise HTTPException(
            status_code=400, 
            detail="User ID header (X-User-ID) is required"
        )
    
    try:
        # Check if user exists and has data
        if x_user_id not in user_rags:
            raise HTTPException(
                status_code=400,
                detail="No data found. Please ingest some URLs first."
            )
            
        rag = get_user_rag(x_user_id)
        response = await rag.query(request.query)
        return {
            "status": "success",
            "answer": response,
            "user_id": x_user_id
        }
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail="No data found. Please ingest some URLs first."
        )
    except Exception as e:
        logger.error(f"Query failed for user {x_user_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Query failed: {str(e)}"
        )

@app.delete("/clear/{user_id}")
async def clear_user_data(user_id: str):
    """Clear a user's RAG data"""
    if user_id in user_rags:
        try:
            user_rags[user_id] = None
            del user_rags[user_id]
            return {"status": "success", "message": f"Cleared data for user {user_id}"}
        except Exception as e:
            logger.error(f"Failed to clear data for user {user_id}: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to clear user data: {str(e)}"
            )
    raise HTTPException(
        status_code=404,
        detail=f"No data found for user {user_id}"
    )

@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint to verify server status"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "active_users": len(user_rags)
    }

@app.get("/operation-status")
async def operation_status():
    """Dummy endpoint to prevent 404 errors"""
    return {"status": "OK"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
