from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, List
from pydantic import BaseModel
import time
import logging
from typing import Optional

# Setup standard logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="MultiModal AI Inference API", version="1.0.0")

# Security Middleware (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory Rate Limiting (Token Bucket approach)
clients = {}
RATE_LIMIT = 50  # Requests per minute
RATE_LIMIT_RESET = 60  # seconds

def check_rate_limit(request: Request):
    client_ip = request.client.host
    current_time = time.time()
    
    if client_ip not in clients:
        clients[client_ip] = {"count": 1, "reset_time": current_time + RATE_LIMIT_RESET}
    else:
        if current_time > clients[client_ip]["reset_time"]:
            clients[client_ip] = {"count": 1, "reset_time": current_time + RATE_LIMIT_RESET}
        else:
            clients[client_ip]["count"] += 1
            if clients[client_ip]["count"] > RATE_LIMIT:
                raise HTTPException(status_code=429, detail="Too Many Requests")
    return True

class InferenceRequest(BaseModel):
    task: str
    context: Optional[str] = ""

class AgentChatRequest(BaseModel):
    message: str

@app.get("/health")
async def health_check():
    return {"status": "ok", "timestamp": time.time()}

@app.post("/api/v1/inference/agent")
async def run_agent(request: AgentChatRequest, rate_limit: bool = Depends(check_rate_limit)):
    """Run the Agentic runtime synchronously."""
    try:
        from agent.agent_v2 import OmniCoreAgent
        
        agent = OmniCoreAgent()
        
        start_time = time.time()
        result = agent.run(task=request.message, verbose=False)
        latency = time.time() - start_time
        
        if "error" in result:
             raise HTTPException(status_code=500, detail=result["error"])
             
        return {
            "response": result.get("response"),
            "latency_seconds": round(latency, 2),
            "trace_id": result.get("trace", {}).get("trace_id", "unknown")
        }
    except Exception as e:
        logger.error(f"Inference failure: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
