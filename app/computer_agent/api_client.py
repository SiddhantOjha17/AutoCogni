# client/computer_agent/api_client.py

import httpx
from typing import Dict, Any
import base64

API_URL = "http://localhost:8000/v1/agent/execute"

async def send_cycle_to_api(
    session_id: str,
    goal: str,
    # screenshot_bytes: bytes
) -> Dict[str, Any]:
    """
    Sends the agent's state to the server and gets the next action.
    """
    # files = {"screenshot": ("screenshot.png", screenshot_bytes, "image/png")}
    data = {"session_id": session_id, "goal": goal}

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            print("Sending data to API...")
            # response = await client.post(API_URL, files=files, data=data)
            response = await client.post(API_URL, data=data)
            response.raise_for_status()
            return response.json()
        except httpx.RequestError as e:
            print(f"API request failed: {e}")
            return {"action": "fail", "thought": f"API connection error: {e}"}

VALIDATION_URL = "http://localhost:8000/v1/agent/validate_action"

async def validate_step_with_api(
    screenshot_bytes: bytes,
    desired_outcome: str
) -> Dict[str, Any]:
    """
    Sends a screenshot and desired outcome to the validation endpoint.
    """
    b64_string = base64.b64encode(screenshot_bytes).decode('utf-8')
    payload = {
        "screenshot_base64": b64_string,
        "desired_outcome": desired_outcome
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            print("🔬 Validating action...")
            response = await client.post(VALIDATION_URL, json=payload)
            response.raise_for_status()
            return response.json()
        except httpx.RequestError as e:
            print(f"Validation request failed: {e}")
            return {"validated": False, "reasoning": f"API connection error: {e}"}