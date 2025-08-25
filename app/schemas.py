# server/app/schemas.py

from pydantic import BaseModel, Field
from typing import Optional, Tuple, Literal, List, Dict, Any

# --- Main Agent Schemas ---


class ActionDetail(BaseModel):
    """Represents a single, executable action with its parameters."""
    tool:str
    parameters: Dict[str, Any]
    desired_outcome: Optional[str] = None
    coordinates: Optional[Tuple[int, int]] = None
    text_to_type: Optional[str] = None
    scroll_direction: Optional[Literal["up", "down"]] = None
    url: Optional[str] = None 

class AgentRequest(BaseModel):
    session_id: str
    goal: str
    # screenshot_base64: str

class AgentResponse(BaseModel):
    thought: str
    actions: List[ActionDetail]

class AgentResponseData(BaseModel):
    """Represents the detailed state returned for an in-progress cycle."""
    thought: Optional[str] = None
    plan: Optional[list] = None
    # Include any other fields your client might need
    vision_analysis: Optional[str] = None


class AgentCycleResponse(BaseModel):
    """The structured response sent back to the client after each cycle."""
    status: str # Will be "completed" or "in_progress"
    output: Optional[str] = None # The final answer when completed
    data: Optional[AgentResponseData] = None # The agent's plan when in_progress

# --- Action Validation Schemas ---

class ValidationRequest(BaseModel):
    """Request to validate a single action's outcome."""
    screenshot_base64: str
    desired_outcome: str

class ValidationResponse(BaseModel):
    """Response from the validation endpoint."""
    validated: bool
    reasoning: str