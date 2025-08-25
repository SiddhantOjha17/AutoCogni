# client/computer_agent/cli.py

import click
import asyncio
import time
import uuid
import json
from os_tools import ScreenController, InputController
from api_client import send_cycle_to_api, validate_step_with_api
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
    """The main asynchronous loop for the agent."""
    session_id = str(uuid.uuid4())
    screen_controller = ScreenController()
    input_controller = InputController()
    goal = task_data.get("goal", "No goal specified.")
    print(f"Starting agent for goal: '{goal}'")

    # if task_data.get("website") and task_data["website"].get("url"):
    #     url = task_data["website"]["url"]
    #     await browser_controller.launch()
    #     await browser_controller.navigate(url=url)
    #     print(f"Waiting for page to load...")
    # else:
    #     await browser_controller.launch()
    #     print("No starting website specified. Agent will begin from the current screen.")
    
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

                if action_type == "navigate":
                    url = action_detail.get("url")
                    if url:
                        await browser_controller.navigate(url=url)  # ✅ use singleton

                elif action_type == "click":
                    selector = action_detail.get("selector")
                    if selector:
                        await browser_controller.click(selector=selector)  # ✅

                elif action_type == "type":
                    selector = action_detail.get("selector")
                    text = action_detail.get("text_to_type")
                    if selector and text:
                        await browser_controller.type_text(selector=selector, text=text)  # ✅

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
