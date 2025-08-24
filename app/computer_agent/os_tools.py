# client/computer_agent/os_tools.py

import pyautogui
from pynput.mouse import Controller as MouseController, Button
# Import Key and platform for the navigate function
from pynput.keyboard import Controller as KeyboardController, Key
import platform
import io
import time

class ScreenController:
    """Handles screen-related operations."""
    def capture_and_encode(self) -> bytes:
        print("--- Capturing pyautogui screenshot ---")
        screenshot = pyautogui.screenshot()
        buffer = io.BytesIO()
        screenshot.save(buffer, format="PNG")
        print("--- Screenshot captured ---")
        return buffer.getvalue()

class InputController:
    """Handles mouse and keyboard operations."""
    def __init__(self):
        self.mouse = MouseController()
        self.keyboard = KeyboardController()
        # Detect the OS for correct keyboard shortcuts
        self.command_key = Key.cmd if platform.system() == "Darwin" else Key.ctrl

    def click(self, x: int, y: int):
        print(f"--- Clicking at ({x}, {y}) ---")
        self.mouse.position = (x, y)
        self.mouse.click(Button.left, 1)
        print("--- Click complete ---")

    def type_text(self, text: str):
        print(f"--- Typing: '{text}' ---")
        self.keyboard.type(text)
        print("--- Typing complete ---")

    def scroll(self, direction: str):
        scroll_amount = 10
        dy = -scroll_amount if direction == "up" else scroll_amount
        print(f"--- Scrolling {direction} ---")
        self.mouse.scroll(0, dy)
        print("--- Scroll complete ---")
        
    def navigate(self, url: str):
        """Navigates to a URL using keyboard shortcuts."""
        print(f"--- Navigating to URL via OS: {url} ---")
        # Press Cmd+L or Ctrl+L to focus the address bar
        with self.keyboard.pressed(self.command_key):
            self.keyboard.press('l')
            self.keyboard.release('l')
        
        time.sleep(0.5) # Wait for the address bar to be focused
        self.keyboard.type(url)
        time.sleep(0.5)
        self.keyboard.press(Key.enter)
        self.keyboard.release(Key.enter)
        print("--- OS Navigation complete ---")

# Instantiate controllers for use in the graph
screen_controller = ScreenController()
input_controller = InputController()