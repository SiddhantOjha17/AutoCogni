# client/computer_agent/cli.py

import click
import asyncio
import time
import uuid
import json
# from os_tools import ScreenController, InputController
from api_client import send_cycle_to_api
from app.computer_agent.playwright_tools import browser_controller 


def main(task_file: str):
    """
    Starts the agentic loop based on a JSON task file.
    """
    try:
        with open(task_file, 'r') as f:
            task_data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error reading or parsing task file: {e}")
        return

    asyncio.run(agent_loop(task_data))

async def agent_loop(task_data: dict):
    """The main asynchronous loop for the agent client."""
    session_id = str(uuid.uuid4())
    goal = task_data.get("goal", "No goal specified.")
    print(f"Starting agent for goal: '{goal}'")

    cycle_count = 0
    while True:
        cycle_count += 1
        print(f"\n--- Cycle {cycle_count} ---")
        try:
            # The client's only job is to trigger the next cycle on the server.
            # The server now handles its own state, including screenshots.
            response = await send_cycle_to_api(session_id, goal)
            
            # --- FIX: Parse the new, structured API response ---
            status = response.get("status")
            
            if status == "completed":
                output = response.get("output", "No final output provided.")
                print(f"\nGoal Achieved! Final Answer: {output}")
                break # Exit the while loop
            
            elif status == "in_progress":
                data = response.get("data", {})
                thought = data.get("thought", "No thought provided.")
                plan = data.get("plan", [])
                print(f"Agent Thought: {thought}")
                if plan:
                    print("Next Plan:")
                    for i, action in enumerate(plan):
                        print(f"  - Step {i+1}: {action.get('tool')} with params {action.get('parameters')}")
                # Wait before triggering the next cycle.
                await asyncio.sleep(2)
            
            else:
                # If the server sends an unexpected status or no status
                print(f"Error: Server returned an unexpected status ('{status}'). Stopping.")
                print(f"Full response: {response}")
                break

        except KeyboardInterrupt:
            print("\nUser interrupted the agent. Exiting.")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            break

    print("\n--- Agent loop finished ---")


if __name__ == "__main__":
    main(task_file="/Users/siddhant/codes/agentic-computer/computer-use/app/computer_agent/task.json")
