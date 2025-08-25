# server/app/agent.py

from app.core.graph import compiled_agent_graph, AgentState
from app.schemas import AgentRequest, AgentResponse
from app.core.memory import memory_manager

async def run_agentic_cycle(request: AgentRequest) -> dict:
    """
    Orchestrates a single cycle of the agentic workflow and returns the
    complete final state dictionary.
    """
    print(f"--- Starting Agent Cycle for Session: {request.session_id} ---")
    
    relevant_memories = memory_manager.search_memory(
        session_id=request.session_id,
        query=request.goal
    )
    
    initial_state = AgentState(
        main_goal=request.goal,
        session_id=request.session_id,
        history=relevant_memories, # Seed history with past experiences
    )
    
    # 1. Execute the graph and get the final state
    final_state = await compiled_agent_graph.ainvoke(initial_state)
    
    # 2. Extract information for memory (if available)
    thought = final_state.get("thought", "No thought generated.")
    plan = final_state.get("concrete_plan", [])
    
    # 3. Store the outcome of this cycle in memory
    memory_entry = (
        f"Goal: {request.goal}\n"
        f"Thought: {thought}\n"
        f"Plan: {[action.tool for action in plan]}" # Log the tools used
    )
    memory_manager.add_memory(request.session_id, memory_entry)

    print(f"--- Agent Cycle Finished for Session: {request.session_id} ---")
    
    # 4. Return the entire state dictionary
    return final_state
