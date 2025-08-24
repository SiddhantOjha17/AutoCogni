from fastapi import APIRouter, UploadFile, File, Form, Depends
from .models import AgentActionResponse
from ..services.agent_service import AgentService, agent_service

# Create an API router
router = APIRouter(
    prefix="/v1/agent",
    tags=["Agentic Workflow"]
)

@router.post("/execute", response_model=AgentActionResponse)
async def execute_agent_cycle(
    session_id: str = Form(...),
    main_goal: str = Form(...),
    screenshot: UploadFile = File(...),
    service: AgentService = Depends(lambda: agent_service) # Dependency injection
):
    """
    Processes a single cycle of the agentic loop.

    Receives the current state (via screenshot and goals) and determines
    the next action to be taken by the client.
    """

    # Read screenshot bytes
    screenshot_bytes = await screenshot.read()
    
    # Delegate the core logic to the agent service
    response = service.process_cycle(
        session_id=session_id,
        main_goal=main_goal,
        screenshot_bytes=screenshot_bytes
    )

    return response