# server/app/core/models.py

import httpx
import base64
from typing import List, Optional, Literal
from openai import OpenAI


class ModelManager:
    """
    Manages API calls to OpenAI (default) or Ollama (if specified).
    """
    def __init__(self, provider: Optional[Literal["openai", "ollama"]] = None, host: str = "http://localhost:11434"):
        self.provider = provider or "openai"
        print(f"--- Initializing ModelManager with provider={self.provider} ---")

        if self.provider == "ollama":
            self.client = httpx.AsyncClient(base_url=host, timeout=120.0)
            self.vision_model_id = "qwen2.5vl:7b"
            self.llm_model_id = "llama3:8b"
        else:  
            self.client = OpenAI()
            self.vision_model_id = "gpt-5-nano-2025-08-07" 
            self.llm_model_id = "gpt-5-nano-2025-08-07"

        print("--- ModelManager initialized ---")

    def _extract_openai_text(self, response) -> str:
        """
        Extracts plain text from OpenAI Response object.
        """
        try:
            for item in response.output:
                if hasattr(item, "content") and item.content:
                    for c in item.content:
                        if getattr(c, "type", None) == "output_text":
                            return c.text.strip()
            return "Error: No text found in OpenAI response."
        except Exception as e:
            print(f"Error extracting OpenAI text: {e}")
            return "Error: Failed to parse OpenAI response."

    async def close(self):
        if self.provider == "ollama":
            await self.client.aclose()

    async def analyze_screenshot(self, screenshot_base64: str, goal: str) -> str:
        """
        Analyzes a screenshot using a vision model.
        """

        prompt = (
            f"You are a computer vision assistant. Your task is to analyze the provided screenshot to identify "
            f"key interactive UI elements relevant for achieving the goal: '{goal}'. "
            f"Describe interactive elements like buttons, input fields, and links. "
            f"For each element, provide a concise description and its bounding box coordinates as [x1, y1, x2, y2]. "
            f"Format each element on a new line. Example: 'Element: Search input field, Coords: [450, 300, 650, 340]'"
        )

        if self.provider == "ollama":
            gen_payload = {
                "model": self.vision_model_id,
                "prompt": prompt,
                "images": [screenshot_base64],
                "stream": False
            }
            try:
                response = await self.client.post("/api/generate", json=gen_payload)
                response.raise_for_status()
                response_json = response.json()
                print(response_json)
                return response_json.get("response", "").strip()
            except httpx.RequestError as e:
                print(f"Error calling Ollama API for vision analysis: {e}")
                return "Error: Could not get vision analysis from Ollama."
        
        else:  # OpenAI
            try:
                response = self.client.responses.create(
                    model=self.vision_model_id,
                    input=[
                        {
                            "role": "user",
                            "content": [
                                { "type": "input_text", "text": prompt },
                                { "type": "input_image", "image_url": f"data:image/png;base64,{screenshot_base64}" }
                            ]
                        }
                    ]
                )
                print("\n",self._extract_openai_text(response),"\n")
                return self._extract_openai_text(response)

            except Exception as e:
                print(f"Error calling OpenAI API for vision analysis: {e}")
                return "Error: Could not get vision analysis from OpenAI."


    async def generate_thought_and_action(self, goal: str, screen_description: str, history: List[str], previous_error: Optional[str] = None) -> str:
        """Generates a plan using abstract tools with improved prompting."""
        
        system_prompt = (
            "You are an expert AI agent controlling a computer. Your primary task is to break down a main goal into smaller, "
            "logical steps. Your response MUST be a single, valid JSON object."
            "\n\n"
            "This JSON object must contain three keys:"
            "\n1. 'thought': A brief, high-level analysis of the current situation and your reasoning for the next step."
            "\n2. 'intermediate_goal': A clear and concise description of the specific objective for this single turn."
            "\n3. 'plan': A list of one or more tool calls to achieve the intermediate goal. Group actions together when they logically follow one another."
            "\n\n"
            "Follow all instructions and rules precisely."
        )
                
        user_prompt = f"""
        **Main User Goal:** "{goal}"

        **Current Screen Analysis:**
        {screen_description}

        **History of Actions Taken:**
        {history}

        **Error from Previous Step:**
        {previous_error if previous_error else 'None'}

        **Guiding Principles:**
        1.  **Analyze:** Review the goal, screen, history, and any errors to understand the current situation.
        2.  **Strategize:** Define the single most logical next step as the `intermediate_goal`."
        3.  **Plan:** Create a `plan` with a list of actions to achieve the `intermediate_goal`.
        4.  **Efficiency is Key:** If multiple actions can be performed sequentially without needing new visual information (e.g., typing a username, then a password, then clicking login), group them into a single plan.
        5.  **Conclude:** If the main goal is visibly achieved on the screen, your plan must use the `finish_task` tool.

        **Available Tools:**

        * ** Tools:**
            * `navigate(url: str)`: Navigates the browser to a specific URL.
            * `click(selector: str)`: Clicks the HTML element matching the CSS selector.
            * `type_text(selector: str, text: str)`: Types text into the HTML element matching the CSS selector.
            * `finish_task(result: str)`: Ends the task. The `result` parameter **MUST** contain the specific information extracted from the screen that answers the main goal. Do not just describe the screen; provide the actual data.

        **Your Task:**
        Generate a JSON object with 'thought', 'intermediate_goal', and 'plan' keys. The plan should be a list of one or more actions.
        **Example of a Multi-Step Plan:**
        {{
            "thought": "The login page is visible. I need to fill in the username and password fields and then click the login button.",
            "intermediate_goal": "Log into the account.",
            "plan": [
                {{
                    "tool": "type_text",
                    "parameters": {{ "selector": "#username", "text": "my_user" }}
                }},
                {{
                    "tool": "type_text",
                    "parameters": {{ "selector": "#password", "text": "my_password" }}
                }},
                {{
                    "tool": "click",
                    "parameters": {{ "selector": "button[type='submit']" }}
                }}
            ]
        }}
        """        

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        if self.provider == "ollama":
            payload = {"model": self.llm_model_id, "messages": messages, "format": "json", "stream": False}
            try:
                response_json = await self._ollama_post_request("/api/chat", payload)
                return response_json.get("message", {}).get("content", "").strip()
            except RuntimeError as e:
                return f'{{ "thought": "Error from Ollama.", "plan": [] }}'
        
        else: # OpenAI
            try:
                response = self.client.chat.completions.create(
                    model=self.llm_model_id,
                    messages=messages,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                return f'{{ "thought": "Error from OpenAI.", "plan": [] }}'


    async def validate_completion(self, goal: str, screen_description: str) -> str:
        """
        Validates if the goal has been met based on the final screen analysis.
        """
        system_prompt = (
            "You are a meticulous validation agent. Your task is to critically assess if the user's goal has been achieved based on the final screen's description. "
            "Be skeptical. Do not assume completion. "
            "Your response MUST be a single JSON object with two keys: 'is_complete' (boolean) and 'reasoning' (a string explaining your decision)."
        )

        user_prompt = f"""
        **User Goal:** "{goal}"

        **Final Screen Analysis:**
        {screen_description}

        **Your Task:**
        Has the goal been successfully and completely achieved?

        **Example Response (Success):**
        {{
            "is_complete": true,
            "reasoning": "The screen clearly shows a metal stool on amazon, which matches the user's goal."
        }}

        **Example Response (Failure):**
        {{
            "is_complete": false,
            "reasoning": "The search results are on the screen, but the user still needs to click on the link that contains the actual weather information."
        }}
        """

        payload = {
            "model": self.llm_model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.0}
        }

        combined_prompt = f"{system_prompt}\n\n{user_prompt}"

        if self.provider == "ollama":
            gen_payload = {
                "model": self.llm_model_id,
                "prompt": combined_prompt,
                "stream": False,
                "options": { "temperature": 0.0 }
            }
            try:
                response = await self.client.post("/api/generate", json=gen_payload)
                response.raise_for_status()
                response_json = response.json()
                print(response_json)
                return response_json.get("response", "").strip()
            except httpx.RequestError as e:
                print(f"Error calling Ollama API for language model: {e}")
                return "FINISH(\"Error: Could not get next action from Ollama.\")"
        
        else:  # OpenAI
            try:
                response = self.client.responses.create(
                    model=self.llm_model_id,
                    input=combined_prompt
                )
                print("\n",self._extract_openai_text(response),"\n")
                return self._extract_openai_text(response)
            except Exception as e:
                print(f"Error calling OpenAI API for language model: {e}")
                return "FINISH(\"Error: Could not get next action from OpenAI.\")"

    async def validate_action_outcome(self, screenshot_base64: str, desired_outcome: str) -> str:
        """Uses the vision model to validate if a screenshot matches a desired outcome."""
        prompt = f"You are a meticulous validation assistant. Does this screenshot reflect the desired outcome? Desired outcome: '{desired_outcome}'. Respond with a single JSON object: {{\"validated\": true/false, \"reasoning\": \"...\"}}."
        payload = {"model": self.vision_model_id, "messages": [{"role": "user", "content": prompt, "images": [screenshot_base64]}], "format": "json", "stream": False}

        if self.provider == "ollama":
            response = await self.client.post("/api/chat", json=payload)
            response.raise_for_status()
            return response.json().get("message", {}).get("content", "").strip()
        else:  # OpenAI
            response = self.client.responses.create(
                model=self.vision_model_id,
                input=[
                    {
                        "role": "user",
                        "content": [
                            { "type": "input_text", "text": prompt },
                            { "type": "input_image", "image_url": f"data:image/png;base64,{screenshot_base64}" }
                        ]
                    }
                ],
                format="json"
            )
            print(self._extract_openai_text(response))

            return self._extract_openai_text(response)


# Default = OpenAI (no provider specified)
model_manager = ModelManager()
