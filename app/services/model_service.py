import httpx
import base64
import json

class ModelService:
    """
    A service to manage and run inference by making API calls to an Ollama server.
    """
    def __init__(self, host: str = "http://localhost:11434"):
        print("Initializing ModelService to connect to Ollama...")
        self.client = httpx.AsyncClient(base_url=host, timeout=60.0)
        self.vision_model_id = "qwen2.5-vl:7b-instruct"
        self.llm_model_id = "llama3:8b"
        print("ModelService initialized.")

    async def close(self):
        """Closes the HTTP client."""
        await self.client.aclose()

    async def get_vision_analysis(self, image_bytes: bytes, goal: str) -> str:
        """
        Analyzes an image using the Qwen-VL model served by Ollama.
        """
        # Encode the image in base64
        base64_image = base64.b64encode(image_bytes).decode('utf-8')

        prompt_text = (
            f"You are a computer vision assistant. Your task is to analyze the provided screenshot to identify "
            f"key UI elements relevant for achieving the goal: '{goal}'. "
            f"Describe interactive elements like buttons, input fields, and links. "
            f"For each element, provide a concise description and its approximate center coordinates as (x, y). "
            f"Format each element as: 'Element: [Description], Coords: (X, Y)'"
        )
        
        payload = {
            "model": self.vision_model_id,
            "prompt": prompt_text,
            "images": [base64_image],
            "stream": False
        }

        try:
            response = await self.client.post("/api/generate", json=payload)
            response.raise_for_status()
            response_json = response.json()
            return response_json.get("response", "").strip()
        except httpx.RequestError as e:
            print(f"Error calling Ollama API for vision analysis: {e}")
            return "Error: Could not connect to Ollama server."

    async def get_next_action(self, goal: str, vision_analysis: str) -> str:
        """
        Decides the next action using Llama 3 served by Ollama.
        """
        system_prompt = "You are an AI agent controlling a computer. Decide the single next action to perform."
        user_prompt = f"""
        Current Goal: "{goal}"

        Screen Analysis:
        {vision_analysis}

        Choose one of the following actions and format your response *exactly* as specified, with no additional text or explanation:
        1. CLICK(x, y, "reason for clicking")
        2. TYPE("text to type", "reason for typing")
        3. SCROLL("direction", "reason for scrolling")
        4. FINISH("reason for finishing")
        """
        
        payload = {
            "model": self.llm_model_id,
            "system": system_prompt,
            "prompt": user_prompt,
            "stream": False,
            "options": {
                "temperature": 0.0 # We want deterministic output for actions
            }
        }
        
        try:
            response = await self.client.post("/api/generate", json=payload)
            response.raise_for_status()
            response_json = response.json()
            return response_json.get("response", "").strip()
        except httpx.RequestError as e:
            print(f"Error calling Ollama API for language model: {e}")
            return "Error: Could not connect to Ollama server."


# Instantiate a single instance of the service for the application
model_service = ModelService()