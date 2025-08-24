from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from .models import model_manager
from app.schemas import AgentResponse, ActionDetail
import json
import re

from app.computer_agent.os_tools import screen_controller, input_controller
from app.computer_agent.playwright_tools import browser_controller

class AgentState(TypedDict):
    main_goal: str
    intermediate_goal: Optional[str] 
    screenshot_base64: str
    session_id: str
    history: List[str]
    vision_analysis: str
    abstract_plan_raw: str
    concrete_plan: List[ActionDetail]
    execution_result: Optional[str]
    error_message: Optional[str]
    final_response: Optional[AgentResponse]

# --- Node Implementations ---

async def vision_node(state: AgentState) -> dict:
    print("\n--- Running Vision Node ---")
    analysis = await model_manager.analyze_screenshot(state["screenshot_base64"], state["main_goal"])
    print(f"Vision Analysis Result: {analysis[:150]}...")
    state["history"].append(f"Vision Analysis: {analysis}")
    # CHANGED: Reset intermediate goal and errors for the new cycle
    return {
        "vision_analysis": analysis, 
        "error_message": None, 
        "execution_result": None,
        "intermediate_goal": None 
    }

async def reasoning_node(state: AgentState) -> dict:
    print("\n--- Running Reasoning Node ---")
    
    # CHANGED: The prompt engineering is now even more critical.
    # The LLM must be instructed to return a JSON object with "intermediate_goal" and "plan" keys.
    # Example Prompt Addition: "Based on the main goal and the screen, first define a specific,
    # immediate 'intermediate_goal' for this step. Then, provide a 'plan' as a JSON array of 
    # actions to achieve it. Your entire response must be a single JSON object."

    previous_error = state.get("error_message")
    
    raw_plan_str = await model_manager.generate_thought_and_action(
        goal=state["main_goal"],
        screen_description=state["vision_analysis"],
        history=state["history"],
        previous_error=previous_error 
    )
    print(f"LLM Raw Plan Object: {raw_plan_str}")
    state["history"].append(f"LLM Raw Output: {raw_plan_str}")
    return {"abstract_plan_raw": raw_plan_str}


def plan_validation_node(state: AgentState) -> dict:
    """Parses, validates, and TRANSFORMS the raw plan object from the LLM."""
    print("\n--- Running Plan Validation Node ---")
    raw_plan_object = state["abstract_plan_raw"]
    
    try:
        json_match = re.search(r"```json\n(.*)\n```", raw_plan_object, re.DOTALL)
        clean_plan_str = json_match.group(1) if json_match else raw_plan_object
        parsed_object = json.loads(clean_plan_str)

        if not isinstance(parsed_object, dict) or "intermediate_goal" not in parsed_object or "plan" not in parsed_object:
            raise ValueError("Response is not a JSON object with 'intermediate_goal' and 'plan' keys.")

        intermediate_goal = parsed_object["intermediate_goal"]
        plan_to_validate = parsed_object["plan"]
        
        if not isinstance(plan_to_validate, list):
            raise ValueError("The 'plan' key must contain a list of actions.")

        print(f"Parsed Intermediate Goal: {intermediate_goal}")
        
        validated_actions = []
        # --- NEW: Transformation Logic ---
        for action in plan_to_validate:
            if not isinstance(action, dict) or "tool" not in action or "parameters" not in action:
                raise ValueError("Invalid action format in plan. Expected 'tool' and 'parameters' keys.")

            tool= action['tool']
            
            parameters = action['parameters']

            # 4. Create the ActionDetail object from the transformed data
            validated_actions.append(ActionDetail(tool=tool,
                                                  parameters=parameters))
            
        print(f"Plan Validated and Transformed Successfully: {validated_actions}")
        return {
            "concrete_plan": validated_actions,
            "intermediate_goal": intermediate_goal,
            "error_message": None
        }

    except (json.JSONDecodeError, ValueError, TypeError) as e:
        error_msg = f"Failed to parse or validate plan object: {e}. Raw response was: {raw_plan_object}"
        print(f"ERROR: {error_msg}")
        return {"concrete_plan": [], "error_message": error_msg, "intermediate_goal": None}

    except (json.JSONDecodeError, ValueError) as e:
        error_msg = f"Failed to parse or validate plan object: {e}. Raw response was: {raw_plan_object}"
        print(f"ERROR: {error_msg}")
        # CHANGED: Ensure concrete_plan is empty on failure
        return {"concrete_plan": [], "error_message": error_msg, "intermediate_goal": None}

async def execution_node(state: AgentState) -> dict:
    """
    Executes the validated plan using the available tool controllers.
    This function is asynchronous to support Playwright's async API.
    """
    intermediate_goal = state.get('intermediate_goal', 'N/A')
    print(f"\n--- Running Execution Node for Intermediate Goal: '{intermediate_goal}' ---")
    state['history'].append(f"Executing plan for intermediate goal: {intermediate_goal}")

    plan = state.get("concrete_plan", [])
    if not plan:
        # This can happen if the plan validation fails but the graph continues.
        # It's a safeguard.
        error_msg = "Execution failed: No valid plan was provided."
        print(f"ERROR: {error_msg}")
        state['history'].append(f"ERROR: {error_msg}")
        return {"error_message": error_msg}

    results = []
    for i, action in enumerate(plan):
        tool_name = action.tool
        params = action.parameters
        print(f"Executing action {i+1}/{len(plan)}: {tool_name} with params {params}")

        try:
            # --- TOOL DISPATCH LOGIC ---
            
            # Browser Tools (asynchronous)
            if tool_name == "navigate":
                await browser_controller.launch()
                await browser_controller.navigate(params['url'])
            elif tool_name == "type_text":
                if "selector" in params:
                    await browser_controller.type_text(**params)
                else:
                    # Fallback to OS-level typing if no selector is provided
                    input_controller.type_text(**params)
            elif tool_name == "click":
                if "selector" in params:
                    await browser_controller.click(**params)
                else:
                    # Fallback to OS-level clicking if no selector is provided
                    input_controller.click(**params)
            
            # OS Tools (synchronous)
            elif tool_name == "scroll":
                input_controller.scroll(**params)

            # Control Flow Tools
            elif tool_name == "finish_task":
                print("--- Task finished by agent ---")
                final_output = params.get("result", "Task completed successfully.")
                state['history'].append(f"Task finished with result: {final_output}")
                return {"final_response": AgentResponse(status="completed", output=final_output)}
            
            else:
                raise ValueError(f"Unknown tool: '{tool_name}'")

            results.append(f"Action '{tool_name}' executed successfully.")

        except Exception as e:
            # If any tool fails, stop and report the error.
            error_msg = f"Error executing action '{tool_name}' with params {params}. Reason: {e}"
            print(f"ERROR: {error_msg}")
            state['history'].append(f"ERROR: {error_msg}")
            # Stop execution on first error and report it to the reasoning node.
            return {"error_message": error_msg}

    execution_summary = "\n".join(results)
    print(f"Execution Summary: {execution_summary}")
    state['history'].append(f"Execution successful for intermediate goal '{intermediate_goal}'.")
    return {"execution_result": execution_summary}

# --- Graph Conditional Logic ---
# This part does not need any changes, as it routes based on the presence
# of 'error_message' or a 'finish_task' action, which is still valid.
def should_continue(state: AgentState) -> str:
    """Determines the next step after plan validation."""
    print("\n--- Running Conditional Edge 'should_continue' ---")
    if state.get("error_message"):
        print("Decision: Plan is invalid. Looping back to reasoning.")
        return "reasoning_node"
    
    if any(action.tool == "finish_task" for action in state["concrete_plan"]):
        print("Decision: 'finish_task' action found. Ending execution.")
        return END
        
    print("Decision: Plan is valid. Proceeding to execution.")
    return "execution_node"

# --- Graph Definition ---
# This part also remains unchanged.
def create_agent_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("vision_node", vision_node)
    workflow.add_node("reasoning_node", reasoning_node)
    workflow.add_node("plan_validation_node", plan_validation_node)
    workflow.add_node("execution_node", execution_node)

    workflow.set_entry_point("vision_node")
    
    workflow.add_edge("vision_node", "reasoning_node")
    workflow.add_edge("reasoning_node", "plan_validation_node")
    
    workflow.add_conditional_edges(
        "plan_validation_node",
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