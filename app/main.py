# server/app/main.py

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from contextlib import asynccontextmanager
from app.agent import run_agentic_cycle
from app.core.models import model_manager # Import the manager
import base64
from app.schemas import AgentRequest, AgentResponse, ValidationRequest, ValidationResponse
import json

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles graceful shutdown of the httpx client.
    """
    yield
    print("--- Application Shutdown ---")
    await model_manager.close()

# Initialize FastAPI with the simplified lifespan
app = FastAPI(
    title="Agentic Computer Automation API (Ollama)",
    description="An API to power an agent that can automate computer tasks via Ollama.",
    version="1.2.0",
    lifespan=lifespan
)

@app.post("/v1/agent/execute", response_model=AgentResponse)
async def execute_agent_cycle(
    session_id: str = Form(...),
    goal: str = Form(...),
    screenshot: UploadFile = File(...)
):
    # This endpoint logic remains exactly the same
    try:
        screenshot_bytes = await screenshot.read()
        screenshot_base64 = base64.b64encode(screenshot_bytes).decode('utf-8')

        agent_request = AgentRequest(
            session_id=session_id,
            goal=goal,
            screenshot_base64=screenshot_base64
        )

        response = await run_agentic_cycle(agent_request)
        return response

    except Exception as e:
        print(f"An error occurred in the endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/agent/validate_action", response_model=ValidationResponse)
async def validate_action(request: ValidationRequest):
    """
    Receives a screenshot and a desired outcome, and validates if they match.
    """
    try:
        validation_raw = await model_manager.validate_action_outcome(
            request.screenshot_base64, request.desired_outcome
        )
        validation_json = json.loads(validation_raw)
        return ValidationResponse(**validation_json)
    except Exception as e:
        print(f"An error occurred during validation: {e}")
        return ValidationResponse(validated=False, reasoning=f"Server error during validation: {e}")


@app.get("/", tags=["Health Check"])
async def read_root():
    return {"status": "ok", "message": "Agentic Server is running."}