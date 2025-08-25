from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from .models import model_manager
from app.schemas import AgentResponse, ActionDetail
import json
import re
from loguru import logger
from app.computer_agent.os_tools import screen_controller, input_controller
from app.computer_agent.playwright_tools import BrowserController


class AgentState(TypedDict):
    main_goal: str
    intermediate_goal: Optional[str] 
    screenshot_base64: str
    session_id: str
    history: List[str]
    vision_analysis: str
    browser: Optional[BrowserController] # Stores the persistent browser instance
    abstract_plan_raw: str
    concrete_plan: List[ActionDetail]
    execution_result: Optional[str]
    error_message: Optional[str]
    final_response: Optional[AgentResponse]

# --- Node Implementations ---

async def vision_node(state: AgentState) -> dict:
    print("\n--- Running Vision Node ---")
    analysis = await model_manager.analyze_screenshot(
        state["screenshot_base64"], state["main_goal"]
    )
    print(f"Vision Analysis Result: {analysis[:150]}...")
    state["history"].append(f"Vision Analysis: {analysis}")
    return {
        "vision_analysis": analysis, 
        "error_message": None, 
        "execution_result": None,
        "intermediate_goal": None 
    }

async def reasoning_node(state: AgentState) -> dict:
    print("\n--- Running Reasoning Node ---")
    previous_error = state.get("error_message")

    raw_plan_str = await model_manager.generate_thought_and_action(
        goal=state["main_goal"],
        screen_description=state["vision_analysis"],
        history=state["history"],
        previous_error=previous_error 
    )
    print(f"LLM Raw Plan Object: {raw_plan_str}")
    state["history"].append(f"LLM Raw Output: {raw_plan_str}")

    # Direct parse here (no separate validation node)
    try:
        json_match = re.search(r"```json\n(.*)\n```", raw_plan_str, re.DOTALL)
        clean_plan_str = json_match.group(1) if json_match else raw_plan_str
        parsed_object = json.loads(clean_plan_str)

        intermediate_goal = parsed_object.get("intermediate_goal", "N/A")
        plan_to_validate = parsed_object.get("plan", [])

        validated_actions = []
        for action in plan_to_validate:
            if isinstance(action, dict) and "tool" in action and "parameters" in action:
                validated_actions.append(ActionDetail(tool=action["tool"], parameters=action["parameters"]))

        print(f"Parsed Intermediate Goal: {intermediate_goal}")
        print(f"Plan Parsed Successfully: {validated_actions}")

        return {
            "abstract_plan_raw": raw_plan_str,
            "concrete_plan": validated_actions,
            "intermediate_goal": intermediate_goal,
            "error_message": None,
        }

    except Exception as e:
        error_msg = f"Failed to parse plan: {e}. Raw response was: {raw_plan_str}"
        print(f"ERROR: {error_msg}")
        return {
            "concrete_plan": [],
            "error_message": error_msg,
            "intermediate_goal": None,
        }

async def execution_node(state: AgentState) -> dict:
    """
    Executes a plan of actions using a persistent browser instance stored in the state.
    """
    intermediate_goal = state.get('intermediate_goal', 'N/A')
    logger.info(f"--- Running Execution Node for Intermediate Goal: '{intermediate_goal}' ---")
    state['history'].append(f"Executing plan for intermediate goal: {intermediate_goal}")

    plan = state.get("concrete_plan", [])
    if not plan:
        error_msg = "Execution failed: No valid plan was provided."
        logger.error(error_msg)
        state['history'].append(f"ERROR: {error_msg}")
        return {"error_message": error_msg}

    # Get or create the browser instance from the state
    browser = state.get("browser")
    if not browser:
        logger.info("No active browser found in state. Creating a new one.")
        # Set headless=True for production/server environments
        browser = BrowserController(headless=False)
        await browser.start()
        state["browser"] = browser

    results = []
    try:
        for i, action in enumerate(plan):
            tool_name = action.tool
            params = action.parameters or {}
            logger.info(f"Executing action {i+1}/{len(plan)}: {tool_name} with params {params}")

            # --- TOOL DISPATCH LOGIC ---
            if tool_name == "navigate":
                await browser.navigate(**params)
            elif tool_name == "type_text":
                await browser.type_text(**params)
            elif tool_name == "click":
                await browser.click(**params)
            elif tool_name == "close_browser":
                logger.info("'close_browser' called. Shutting down browser.")
                await browser.stop()
                state["browser"] = None # Remove from state
            elif tool_name == "scroll":
                input_controller.scroll(**params)
            elif tool_name == "type_text_os":
                input_controller.type_text(**params)
            elif tool_name == "click_os":
                input_controller.click(**params)
            elif tool_name == "finish_task":
                logger.info("--- Task finished by agent ---")
                final_output = params.get("result", "Task completed successfully.")
                state['history'].append(f"Task finished with result: {final_output}")
                if browser:
                    await browser.stop()
                    state["browser"] = None
                return {"final_response": AgentResponse(status="completed", output=final_output)}
            else:
                raise ValueError(f"Unknown tool: '{tool_name}'")

            results.append(f"Action '{tool_name}' executed successfully.")

    except Exception as e:
        error_msg = f"Error executing action. Reason: {e}"
        logger.error(error_msg, exc_info=True)
        state['history'].append(f"ERROR: {error_msg}")
        # Clean up browser on critical error
        if browser:
            await browser.stop()
            state["browser"] = None
        return {"error_message": error_msg}

    execution_summary = "\n".join(results)
    logger.info(f"Execution Summary:\n{execution_summary}")
    state['history'].append(f"Execution successful for intermediate goal '{intermediate_goal}'.")
    return {"execution_result": execution_summary}

# --- Graph Definition (no validation node) ---
def should_continue(state: AgentState) -> str:
    if state.get("error_message"):
        return "reasoning_node"
    if any(action.tool == "finish_task" for action in state.get("concrete_plan", [])):
        return END
    return "execution_node"

def create_agent_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("vision_node", vision_node)
    workflow.add_node("reasoning_node", reasoning_node)
    workflow.add_node("execution_node", execution_node)

    workflow.set_entry_point("vision_node")
    workflow.add_edge("vision_node", "reasoning_node")
    workflow.add_conditional_edges(
        "reasoning_node",
        should_continue,
        {
            "execution_node": "execution_node",
            "reasoning_node": "reasoning_node",
            END: END
        }
    )
    workflow.add_edge("execution_node", "vision_node")

    print("--- Agent Graph Compiled ---")
    return workflow.compile()


compiled_agent_graph = create_agent_graph()
