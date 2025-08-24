# client/computer_agent/cli.py

import click
import asyncio
import time
import uuid
import json
from os_tools import ScreenController, InputController
from api_client import send_cycle_to_api, validate_step_with_api
from app.computer_agent.playwright_tools import BrowserController

# MODIFIED: Changed from --goal to --task-file
# @click.command()
# @click.option(
#     "--task-file",
#     type=click.Path(exists=True, dir_okay=False),
#     required=True,
#     help="Path to the JSON file describing the task."
# )

def main(task_file: str):
    """
    Starts the agentic loop based on a JSON task file.
    """
    # NEW: Load and parse the task file
    try:
        with open(task_file, 'r') as f:
            task_data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error reading or parsing task file: {e}")
        return

    asyncio.run(agent_loop(task_data))


async def agent_loop(task_data: dict):
    """The main asynchronous loop for the agent."""
    session_id = str(uuid.uuid4())
    screen_controller = ScreenController()
    input_controller = InputController()
    browser_controller = BrowserController()
    goal = task_data.get("goal", "No goal specified.")
    print(f"Starting agent for goal: '{goal}'")
    # ... (initial navigation logic remains the same, but now it's redundant as the agent can plan it)
    # Let's simplify by letting the agent handle navigation from the start.
    if task_data.get("website") and task_data["website"].get("url"):
        url = task_data["website"]["url"]
        await browser_controller.launch()
        await browser_controller.navigate(url=url)
        print(f"Waiting for page to load...")
    else:
        asyncio.run(browser_controller.launch())
        print("No starting website specified. Agent will begin from the current screen.")
    
    cycle_count = 0
    while True:
        cycle_count += 1
        print(f"\n--- Cycle {cycle_count} ---")
        try:
            screenshot_bytes = screen_controller.capture_and_encode()
            response = await send_cycle_to_api(session_id, goal, screenshot_bytes)
            thought = response.get("thought", "No thought provided.")
            actions = response.get("actions", [])
            print(f"Agent Thought: {thought}")

            if not actions:
                print("Agent failed: No actions received.")
                break

            # Execute the sequence of concrete actions from the server
            for i, action_detail in enumerate(actions):
                action_type = action_detail.get("action_type")
                print(f"  - Action {i+1}/{len(actions)}: {action_type.upper()}")
                
                # NEW: Handle navigate action
                if action_type == "navigate":
                    url = action_detail.get("url")
                    if url:
                        input_controller.navigate(url)

                elif action_type == "click":
                    coords = action_detail.get("coordinates")
                    if coords:
                        input_controller.click(coords[0], coords[1])
                elif action_type == "type":
                    text = action_detail.get("text_to_type")
                    if text:
                        input_controller.type_text(text)
                elif action_type == "finish":
                    print("Goal achieved! Agent is finishing.")
                    return
                elif action_type == "fail":
                    print(f"Agent failed: {thought}")
                    return
                
                # Wait for UI to update after each action
                time.sleep(2)

        except KeyboardInterrupt:
            print("\nUser interrupted the agent. Exiting.")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            break
        
if __name__ == "__main__":
    main(task_file="/Users/siddhant/codes/agentic-computer/computer-use/app/computer_agent/task.json")