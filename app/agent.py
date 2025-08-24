# server/app/agent.py

from app.core.graph import compiled_agent_graph, AgentState
from app.schemas import AgentRequest, AgentResponse
from app.core.memory import memory_manager

async def run_agentic_cycle(request: AgentRequest) -> AgentResponse:
    """
    Orchestrates a single cycle of the agentic workflow.
    """
    print(f"--- Starting Agent Cycle for Session: {request.session_id} ---")
    
    # 1. Retrieve relevant memories (optional, for future enhancement)
    relevant_memories = memory_manager.search_memory(
        session_id=request.session_id,
        query=request.goal
    )
    
    # 2. Prepare the initial state for the graph
    initial_state = AgentState(
        main_goal=request.goal,
        screenshot_base64=request.screenshot_base64,
        session_id=request.session_id,
        history=relevant_memories, # Seed history with past experiences
        # The rest will be populated by the graph
        vision_analysis=None,
        next_action_raw=None,
        final_response=None
    )
    
    # 3. Invoke the LangGraph to process the state
    final_state = await compiled_agent_graph.ainvoke(initial_state)
    
    # 4. Extract the final response
    response = final_state.get("final_response")
    print(response)
    
    if not response:
        # Fallback in case the graph fails unexpectedly
        return AgentResponse(thought="The agent failed to produce a valid action.", actions=[])
        
    # 5. Store the outcome of this cycle in memory
    memory_entry = (
        f"Goal: {request.goal}\n"
        f"Action: {response.actions}\n"
        f"Thought: {response.thought}"
    )
    memory_manager.add_memory(request.session_id, memory_entry)

    print(f"--- Agent Cycle Finished. Action: {response.actions} ---")
    return response