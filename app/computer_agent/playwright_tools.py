# client/computer_agent/playwright_tools.py

import io
from playwright.async_api import async_playwright, Browser, Page, Playwright

class BrowserController:
    """Handles browser-related operations using Playwright."""
    
    def __init__(self):
        self.playwright: Playwright | None = None
        self.browser: Browser | None = None
        self.page: Page | None = None

    async def launch(self):
        """Launches the browser and creates a new page."""
        print("--- Launching Playwright Browser ---")
        self.playwright = await async_playwright().start()
        # Using Chromium, but can be configured for firefox or webkit
        self.browser = await self.playwright.chromium.launch(headless=False) 
        self.page = await self.browser.new_page()
        print("--- Browser Launched Successfully ---")

    async def navigate(self, url: str):
        """Navigates the current page to a URL."""
        if not self.page:
            raise ConnectionError("Browser is not running. Call launch() first.")
        print(f"--- Navigating to URL: {url} ---")
        await self.page.goto(url, wait_until="domcontentloaded")
        print(f"--- Navigation to {url} complete ---")

    async def click(self, selector: str):
        """Clicks on an element specified by a CSS selector."""
        if not self.page:
            raise ConnectionError("Browser is not running. Call launch() first.")
        print(f"--- Clicking element with selector: '{selector}' ---")
        await self.page.click(selector)
        print("--- Click complete ---")

    async def type_text(self, selector: str, text: str):
        """Types text into an element specified by a CSS selector."""
        if not self.page:
            raise ConnectionError("Browser is not running. Call launch() first.")
        print(f"--- Typing '{text}' into element with selector: '{selector}' ---")
        await self.page.fill(selector, text)
        print("--- Typing complete ---")

    async def capture_and_encode(self) -> bytes:
        """Captures a screenshot of the current page and returns it as bytes."""
        if not self.page:
            raise ConnectionError("Browser is not running. Call launch() first.")
        print("--- Capturing Playwright screenshot ---")
        screenshot_bytes = await self.page.screenshot()
        print("--- Screenshot captured ---")
        return screenshot_bytes

    async def close(self):
        """Closes the browser and cleans up resources."""
        if self.browser:
            print("--- Closing Playwright Browser ---")
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
        print("--- Browser Closed ---")

# A single instance to be used by the agent
browser_controller = BrowserController()