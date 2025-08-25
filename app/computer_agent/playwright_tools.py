from loguru import logger
from playwright.async_api import async_playwright, Browser, Page, Playwright, Error as PlaywrightError


class BrowserController:
    """
    Handles browser operations using Playwright with manual start/stop controls
    for persistent sessions.
    """
    
    def __init__(self, headless: bool = True):
        """
        Initializes the controller.
        
        Args:
            headless (bool): Whether to run the browser in headless mode.
        """
        self.playwright: Playwright | None = None
        self.browser: Browser | None = None
        self.page: Page | None = None
        self.headless = headless

    async def start(self):
        """
        Manually starts Playwright, launches the browser, and creates a new page.
        """
        if self.page:
            logger.warning("Browser is already running.")
            return

        logger.info("--- Starting Playwright ---")
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=self.headless,
            args=["--disable-blink-features=AutomationControlled"]
        )
        self.page = await self.browser.new_page(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
        )
        logger.info("--- Browser and page are ready ---")

    async def stop(self):
        """
        Manually closes the browser and stops the Playwright instance.
        """
        if self.browser:
            await self.browser.close()
            logger.info("--- Browser closed ---")
        if self.playwright:
            await self.playwright.stop()
            logger.info("--- Playwright stopped ---")
        self.page = None
        self.browser = None
        self.playwright = None

    def _ensure_browser_is_running(self):
        """Checks if the browser is initialized before performing an action."""
        if not self.page or not self.browser:
            raise RuntimeError("Browser is not started. Please call .start() before using browser actions.")

    async def navigate(self, url: str):
        """Navigates the current page to a URL."""
        self._ensure_browser_is_running()
        try:
            logger.info(f"Navigating to URL: {url}")
            await self.page.goto(url, wait_until="domcontentloaded", timeout=60000)
            logger.info(f"Navigation to {url} complete")
        except PlaywrightError as e:
            logger.error(f"Failed to navigate to {url}: {e}")
            raise

    async def click(self, selector: str, timeout: int = 10000):
        """Waits for and clicks on an element specified by a CSS selector."""
        self._ensure_browser_is_running()
        try:
            logger.info(f"Waiting for and clicking element: '{selector}'")
            await self.page.wait_for_selector(selector, state="visible", timeout=timeout)
            await self.page.click(selector)
            logger.info(f"Successfully clicked element: '{selector}'")
        except PlaywrightError as e:
            logger.error(f"Failed to click selector '{selector}': {e}")
            raise

    async def type_text(self, selector: str, text: str, timeout: int = 10000):
        """Waits for and types text into an element specified by a CSS selector."""
        self._ensure_browser_is_running()
        try:
            logger.info(f"Typing '{text}' into element: '{selector}'")
            await self.page.wait_for_selector(selector, state="visible", timeout=timeout)
            await self.page.fill(selector, text)
            logger.info("Typing complete")
        except PlaywrightError as e:
            logger.error(f"Failed to type into selector '{selector}': {e}")
            raise

    async def capture_and_encode(self) -> bytes | None:
        """Captures a screenshot of the current page and returns it as bytes."""
        self._ensure_browser_is_running()
        try:
            logger.info("Capturing screenshot")
            screenshot_bytes = await self.page.screenshot()
            logger.info("Screenshot captured successfully")
            return screenshot_bytes
        except PlaywrightError as e:
            logger.error(f"Failed to capture screenshot: {e}")
            return None

browser_controller = BrowserController()