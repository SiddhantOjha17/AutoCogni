# server/app/core/memory.py

from mem0 import MemoryClient
from typing import List
from dotenv import load_dotenv
import os

class CloudMemoryManager:
    """
    A wrapper class for the Mem0 Cloud Client.
    This class handles creating, storing, and retrieving agent experiences
    using the managed Mem0 service.
    """
    def __init__(self):
        print("Initializing Cloud Memory Manager (Mem0 Client)...")
        load_dotenv()
        
        api_key = os.getenv("MEM0_API_KEY")
        if not api_key:
            raise ValueError("MEM0_API_KEY not found in environment variables. Please add it to your .env file.")
            
        self.client = MemoryClient(api_key=api_key)

    def add_memory(self, session_id: str, entry: str):
        """Adds a new memory entry for a given session (user)."""
        try:
            messages = [
                {"role": "user", "content": entry}
            ]
            print("Entry: ", messages)
            self.client.add(messages, user_id=session_id)
            print(f"MEMORY [ADD] for Session {session_id}")
        except Exception as e:
            print(f"Error adding memory for session {session_id}: {e}")


    def search_memory(self, session_id: str, query: str) -> List[dict]:
        """Searches for relevant memories for a given session (user)."""
        try:
            # Pass user_id directly as a keyword argument
            results = self.client.search(query=query, user_id=session_id)
            print(f"MEMORY [SEARCH] for Session {session_id} found {len(results)} results.")
            return results
        except Exception as e:
            print(f"Error searching memory for session {session_id}: {e}")
            return []

# Instantiate the new cloud-based manager
memory_manager = CloudMemoryManager()