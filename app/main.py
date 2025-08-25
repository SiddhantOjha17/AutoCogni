# server/app/main.py

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from contextlib import asynccontextmanager
from app.agent import run_agentic_cycle
from app.core.models import model_manager # Import the manager
from app.schemas import AgentRequest, AgentCycleResponse, AgentResponseData

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

@app.post("/v1/agent/execute", response_model=AgentCycleResponse)
async def execute_agent_cycle(
    session_id: str = Form(...),
    goal: str = Form(...)
):
    """
    Executes one cycle of the agent and returns a status indicating
    if the task is complete or in progress.
    """
    try:
        agent_request = AgentRequest(
            session_id=session_id,
            goal=goal,
        )

        # This function should return the final state dictionary from the graph
        agent_state = await run_agentic_cycle(agent_request)
        
        print("Agent State: ", agent_state)

        # --- FIX: Check for the completion signal ---
        if "final_response" in agent_state and agent_state["final_response"]:
            final_output = agent_state["final_response"].get("output", "Task completed successfully.")
            print(f"Agent task completed for session {session_id}. Final output: {final_output}")
            
            # Return a "completed" status to the client
            return AgentCycleResponse(status="completed", output=final_output)
        
        else:
            # If not complete, return an "in_progress" status with the agent's next steps
            response_data = AgentResponseData(
                thought=agent_state.get("thought"),
                plan=agent_state.get("concrete_plan"),
                vision_analysis=agent_state.get("vision_analysis")
            )
            return AgentCycleResponse(status="in_progress", data=response_data)

    except Exception as e:
        print(f"An error occurred in the endpoint: {e}")
        # It's good practice to log the full traceback here
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", tags=["Health Check"])
async def read_root():
    return {"status": "ok", "message": "Agentic Server is running."}