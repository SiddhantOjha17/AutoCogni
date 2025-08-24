import re
from .model_service import model_service
from ..api.models import AgentActionResponse

class AgentService:
    def parse_action_string(self, action_str: str) -> tuple:
        """
        Parses the raw LLM output string into a structured action command.
        (This function remains unchanged)
        """
        try:
            command = action_str.split('(', 1)[0].upper()
            
            if command == "CLICK":
                match = re.search(r'CLICK\(\s*(\d+)\s*,\s*(\d+)\s*,(.*)\)', action_str, re.IGNORECASE)
                if match:
                    x, y, reason = match.groups()
                    return "click", (int(x), int(y)), None, reason.strip().strip('"')
            
            elif command == "TYPE":
                match = re.search(r'TYPE\((.*),(.*)\)', action_str, re.IGNORECASE)
                if match:
                    text_to_type, reason = match.groups()
                    return "type", None, text_to_type.strip().strip('"'), reason.strip().strip('"')

            elif command == "SCROLL":
                match = re.search(r'SCROLL\((.*),(.*)\)', action_str, re.IGNORECASE)
                if match:
                    direction, reason = match.groups()
                    return "scroll", None, direction.strip().strip('"'), reason.strip().strip('"')
            
            elif command == "FINISH":
                match = re.search(r'FINISH\((.*)\)', action_str, re.IGNORECASE)
                if match:
                    reason = match.groups()[0]
                    return "finish", None, None, reason.strip().strip('"')

        except Exception as e:
            print(f"Error parsing action string: {e}")

        return "error", None, None, f"Could not parse model output: {action_str}"


    async def process_cycle(self, session_id: str, main_goal: str, screenshot_bytes: bytes) -> AgentActionResponse:
        """
        Orchestrates a single agent cycle using async calls to the ModelService.
        """
        # 1. Get vision analysis from Qwen-VL via Ollama
        vision_analysis = await model_service.get_vision_analysis(screenshot_bytes, main_goal)
        print(f"--- Vision Analysis ---\n{vision_analysis}\n-----------------------")

        # 2. Get next action from Llama 3 via Ollama
        action_string = await model_service.get_next_action(main_goal, vision_analysis)
        print(f"--- Raw Action ---\n{action_string}\n------------------")

        # 3. Parse the raw action string into a structured response
        action_type, coords, text, thought = self.parse_action_string(action_string)

        return AgentActionResponse(
            session_id=session_id,
            next_action=action_type,
            coordinates=coords,
            text_to_type=text,
            status="success" if action_type != "error" else "error",
            thought=thought
        )

# Instantiate the service
agent_service = AgentService()