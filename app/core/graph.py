from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from .models import model_manager
from app.schemas import AgentResponse, ActionDetail
import json
import re
import base64
from loguru import logger
from app.computer_agent.os_tools import screen_controller, input_controller
from app.computer_agent.playwright_tools import BrowserController


class AgentState(TypedDict):
    main_goal: str
    intermediate_goal: Optional[str] 
    session_id: str
    history: List[str]
    vision_analysis: str
    browser: Optional[BrowserController]
    abstract_plan_raw: str
    concrete_plan: List[ActionDetail]
    execution_result: Optional[str]
    error_message: Optional[str]
    final_response: Optional[dict] # Changed to dict for flexibility

# --- Node Implementations (Your existing nodes are correct and remain unchanged) ---

async def vision_node(state: AgentState) -> dict:
    """
    Captures a screenshot from the appropriate source (browser or desktop),
    encodes it to base64, and sends it for vision analysis.
    """
    print("\n--- Running Vision Node ---")
        
    browser = state.get("browser")
    screenshot_bytes = None
    
    if browser and browser.page:
        logger.info("Capturing screenshot from BROWSER...")
        screenshot_bytes = await browser.capture_and_encode()
    else:
        logger.info("No active browser. Capturing screenshot from DESKTOP...")
        screenshot_bytes = screen_controller.capture_and_encode()

    if not screenshot_bytes:
        error_msg = "Failed to capture screenshot."
        logger.error(error_msg)
        return {"error_message": error_msg}

    screenshot_base64 = base64.b64encode(screenshot_bytes).decode('utf-8')
    
    analysis = await model_manager.analyze_screenshot(
        screenshot_base64, state["main_goal"]
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

    try:
        json_match = re.search(r"```json\n(.*)\n```", raw_plan_str, re.DOTALL)
        clean_plan_str = json_match.group(1) if json_match else raw_plan_str
        parsed_object = json.loads(clean_plan_str)

        intermediate_goal = parsed_object.get("intermediate_goal", "N/A")
        plan_to_validate = parsed_object.get("plan", [])

        validated_actions = [ActionDetail(tool=a["tool"], parameters=a["parameters"]) for a in plan_to_validate]

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
        return {"concrete_plan": [], "error_message": error_msg}

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

    browser = state.get("browser")
    
    is_browser_action = any(a.tool in ["navigate", "click", "type_text"] for a in plan)
    if is_browser_action and not browser:
        logger.info("Browser action detected. Starting browser.")
        browser = BrowserController(headless=False)
        await browser.start()
        state["browser"] = browser

    results = []
    try:
        for action in plan:
            tool_name = action.tool
            params = action.parameters or {}
            logger.info(f"Executing action: {tool_name} with params {params}")

            if tool_name == "finish_task":
                logger.info("--- Task finished by agent ---")
                final_output = params.get("result", "Task completed.")
                if browser:
                    await browser.stop()
                    state["browser"] = None

                return {"final_response": {"status": "completed", "output": final_output}}
            
            elif tool_name == "navigate": await browser.navigate(**params)
            elif tool_name == "type_text": await browser.type_text(**params)
            elif tool_name == "click": await browser.click(**params)
            elif tool_name == "scroll": input_controller.scroll(**params)
            else:
                raise ValueError(f"Unknown tool: {tool_name}")

            results.append(f"Action '{tool_name}' executed successfully.")

    except Exception as e:
        error_msg = f"Error executing action. Reason: {e}"
        logger.error(error_msg, exc_info=True)
        if browser:
            await browser.stop()
            state["browser"] = None
        return {"error_message": error_msg}

    execution_summary = "\n".join(results)
    state['history'].append(f"Execution successful for intermediate goal '{intermediate_goal}'.")
    return {"execution_result": execution_summary}

def should_loop_or_end(state: AgentState) -> str:
    """
    Decision node to determine if the agent should continue or end the task.
    This runs AFTER the execution node.
    """
    if state.get("final_response"):
        return END
    else:
        return "vision_node"

def create_agent_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("vision_node", vision_node)
    workflow.add_node("reasoning_node", reasoning_node)
    workflow.add_node("execution_node", execution_node)

    workflow.set_entry_point("vision_node")
    
    workflow.add_edge("vision_node", "reasoning_node")
    workflow.add_edge("reasoning_node", "execution_node")

    workflow.add_conditional_edges(
        "execution_node",
        should_loop_or_end,
        {
            "vision_node": "vision_node",
            END: END
        }
    )

    print("--- Agent Graph Compiled ---")
    return workflow.compile()


compiled_agent_graph = create_agent_graph()
