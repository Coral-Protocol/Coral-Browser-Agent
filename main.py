import urllib.parse
import os
import json
import asyncio
import logging
from collections import deque
from typing import List, Dict
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor
from utils.manual_input import get_user_input
from utils.coral_tools import initialize_agent, wait_for_mentions, process_and_respond

MANUAL_INPUT = False
class BrowserAgent:
    """Manages server connection, tools, and agent execution with enhanced history and error handling."""

    def __init__(self, history_maxlen: int = 5):
        self.logger = logging.getLogger(__name__)
        self.history = deque(maxlen=history_maxlen)
        self._initialize_logging()
        self._initialize()

    def _initialize_logging(self):
        """Set up logging with a specific format."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    def _validate_env_vars(self) -> None:
        """Validate required environment variables."""
        required_vars = [
            "MODEL_NAME",
            "MODEL_PROVIDER",
            "MODEL_API_KEY"
        ]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            self.logger.error(f"Missing environment variables: {', '.join(missing_vars)}")
            raise SystemExit("Initialization failed due to missing environment variables")

    def _initialize(self):
        """Initialize client and validate environment variables."""
        runtime = os.getenv("CORAL_ORCHESTRATION_RUNTIME", None)
        if runtime is None:
            load_dotenv()
        self._validate_env_vars()

        self.client = MultiServerMCPClient(
            connections={
                "playwright": {
                    "transport": "stdio",
                    "command": "npx",
                    "args": ["@playwright/mcp@latest"]
                }
            }
        )

    def _format_history(self) -> str:
        """Format chat history as a string with user and assistant messages."""
        if not self.history:
            return "None"
        formatted = []
        for i, (user_input, assistant_output) in enumerate(self.history, 1):
            formatted.append(f"{i}. User: {user_input}\n   Assistant: {assistant_output}")
        return "\n".join(formatted)

    def _get_tools_description(self, tools: List) -> str:
        """Format tools description for logging (not included in prompt)."""
        return "\n".join(f"Tool: {tool.name}, Schema: {json.dumps(tool.args)}" for tool in tools)

    async def create_agent(self, agent_tools: List) -> AgentExecutor:
        """Create LangChain agent with optimized prompt."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an autonomous web browser agent designed to perform complex tasks by simulating human-like browsing. Your goal is to complete the given task efficiently and accurately. Follow these guidelines strictly:

        - **Planning**: Before taking any action, plan your steps in detail. Break down the task into sequential sub-tasks. Think step-by-step about what needs to be done, which tools to use, and why. Use the chat history to inform your planning and avoid repeating actions.

        - **Page State Management**: Always use `browser_snapshot` before any interaction (e.g., clicking, typing, filling forms) to get the current page state and obtain fresh element references (refs). Refs from old snapshots may become stale, leading to errors. If an action fails due to a stale ref or element not found, immediately take a new snapshot and retry with updated refs.

        - **Element Inspection**: Carefully examine the page snapshot to determine the actual HTML structure of elements. For example, a 'combobox' in the snapshot might be an <input> with role='combobox' (autosuggest) rather than a <select>. Choose tools accordingly—do not assume all comboboxes are traditional dropdowns.

        - **Navigation and Loading**: Use `browser_navigate` to go to URLs. After navigation or any action that changes the page (e.g., submit), use `browser_wait_for` to ensure the page has loaded fully or specific text/elements appear/disappear. Close pop-ups or overlays by identifying them in the snapshot (e.g., look for 'close' buttons, modals, or banners) and using `browser_click` on close buttons.

        - **Cookies and Dialogs**: Upon loading a page, check the snapshot for cookie consent banners or prompts. Automatically accept cookies if prompted by locating the 'Accept' or 'Agree' button in the snapshot and using `browser_click` on it. Handle alerts, confirms, or prompts with `browser_handle_dialog`. If console messages indicate issues (use `browser_console_messages`), address them.

        - **CAPTCHAs and Challenges**: If a CAPTCHA appears (detect via snapshot or console errors), use `browser_take_screenshot` to capture it. For text-based CAPTCHAs, solve them if possible using logic or `browser_evaluate`. For image-based, describe the image (after screenshot) and attempt to solve via appropriate tools or manual simulation. If unsolvable, note it and try alternative paths.

        - **Form Filling**: Use `browser_fill_form` for filling multiple fields at once, specifying field types accurately (e.g., textbox, checkbox, radio, combobox, slider). Verify from snapshot if a combobox is a <select> (use 'combobox' type or `browser_select_option`) or an autosuggest input (fallback to `browser_type` to enter text, `browser_wait_for` suggestions to appear, then `browser_click` on the desired option or use `browser_press_key` like ArrowDown and Enter). If `browser_fill_form` fails (e.g., not a select element), retry with alternative tools like `browser_type`.

        - **Typing and Input**: Use `browser_type` for typing into fields, especially if slow typing is needed to trigger events. Set 'slowly' to true for character-by-character input. Use `browser_press_key` for special keys like Enter, Tab, or arrows.

        - **Clicking and Interactions**: Use `browser_click` for buttons/links, providing human-readable descriptions and exact refs from snapshot. For double-click or right-click, set parameters accordingly. Use `browser_hover` if needed to reveal elements. For drag-and-drop, use `browser_drag`.

        - **File Uploads**: Use `browser_file_upload` with absolute paths to files.

        - **Screenshots and Debugging**: Use `browser_take_screenshot` for visual verification, especially for CAPTCHAs or to confirm states. Use `browser_console_messages` to check for errors, `browser_network_requests` for API calls, and `browser_evaluate` for custom JavaScript execution on elements.

        - **Tabs Management**: Use `browser_tabs` to handle multiple tabs if the task requires it (e.g., open new tab for background checks).

        - **Error Handling**: If an action fails (e.g., element not found, invalid input, or tool-specific errors like 'not a select element'):
        1. Take a new `browser_snapshot` to refresh refs and assess current state.
        2. Check `browser_console_messages` for clues.
        3. If page seems unresponsive, try navigating back with `browser_navigate_back` or reloading (simulate reload via `browser_navigate` to current URL if needed).
        4. Analyze the error from tool output or console messages.
        5. Adjust your approach: e.g., use alternative elements, wait longer, or switch tools (e.g., from `browser_fill_form` to `browser_type` for non-standard inputs).
        6. For rejected inputs (e.g., invalid postcode), verify field requirements from page text or labels, and retry with corrected values.
        - Never repeat the exact same failed action without changes. Escalate persistent issues by noting them in your reasoning.

        - **Efficiency**: Minimize unnecessary actions. Use `browser_resize` only if viewport size affects layout. Avoid infinite loops by tracking progress in your scratchpad.

        - **Context**: Maintain context from chat history. If the task involves dynamic content, use `browser_wait_for` liberally.

        Task: {input_query}
        Chat History: {history}
        Tools Description: {tools_description}
        """),
            ("placeholder", "{agent_scratchpad}")
        ])

        try:
            model = init_chat_model(
                model=os.getenv("MODEL_NAME"),
                model_provider=os.getenv("MODEL_PROVIDER"),
                api_key=os.getenv("MODEL_API_KEY"),
                temperature=float(os.getenv("MODEL_TEMPERATURE", 0.0)),
                max_tokens=int(os.getenv("MODEL_MAX_TOKENS", 8000)),
                base_url=os.getenv("MODEL_BASE_URL", None)
            )
            agent = create_tool_calling_agent(model, agent_tools, prompt)
            return AgentExecutor(agent=agent, tools=agent_tools, verbose=True)
        except Exception as e:
            self.logger.error(f"Failed to create agent: {str(e)}")
            raise

    async def collect_inputs(self, input_queue: asyncio.Queue, logger1=None, client1=None, agent_tools1=None):
        """Collect inputs and add to queue, responding if agent is busy."""
        while True:
            try:
                if MANUAL_INPUT:
                    input_query, should_exit = get_user_input(self.logger)
                    if should_exit:
                        self.logger.info("Exiting via user input")
                        return
                    if not input_query:
                        await asyncio.sleep(0.1)
                        continue
                    if self.is_busy:
                        print("Bot: Agent is processing previous request and is busy")
                    await input_queue.put((input_query, None, None))
                else:
                    result = await wait_for_mentions(logger1, client1, agent_tools1)
                    if not result:
                        await asyncio.sleep(0.1)
                        continue
                    thread_id, sender_id, input_query = result
                    if not input_query:
                        await asyncio.sleep(0.1)
                        continue
                    if self.is_busy:
                        await process_and_respond(
                            logger1,
                            agent_tools1,
                            "Agent is processing previous request and is busy",
                            thread_id,
                            sender_id
                        )
                    await input_queue.put((input_query, thread_id, sender_id))
            except Exception as e:
                self.logger.error(f"Error collecting input: {str(e)}")
                print("Bot: Error collecting input. Please try again.")
                await asyncio.sleep(0.5)

    async def process_inputs(self, input_queue: asyncio.Queue, agent: AgentExecutor, logger1=None, agent_tools1=None):
        """Process inputs from the queue one at a time."""
        while True:
            try:
                input_query, thread_id, sender_id = await input_queue.get()
                self.is_busy = True
                history_str = self._format_history()
                try:
                    response = await agent.ainvoke(
                        {
                            "input_query": input_query,
                            "history": history_str,
                            "tools_description": self.tools_description,
                            "agent_scratchpad": ""
                        }
                    )
                    output = response.get("output", "No response generated")
                except Exception as e:
                    self.logger.error(f"Agent invocation failed: {str(e)}")
                    output = f"Error: {str(e)}"

                print("Bot:", output)
                self.history.append((input_query, output))

                if not MANUAL_INPUT and thread_id and sender_id:
                    await process_and_respond(logger1, agent_tools1, output, thread_id, sender_id)

                input_queue.task_done()
                self.is_busy = False
            except Exception as e:
                self.logger.error(f"Error processing queued input: {str(e)}")
                self.is_busy = False
                input_queue.task_done()
                await asyncio.sleep(0.5)

    async def run(self):
        """Handle user input loop with persistent session and asynchronous input handling."""
        async with self.client.session("playwright") as session:
            try:
                # Initialize tools and agent
                agent_tools = await load_mcp_tools(session)
                self.tools_description = self._get_tools_description(agent_tools)
                self.logger.info(f"Initialized with {len(agent_tools)} tools")
                print(self.tools_description)

                agent = await self.create_agent(agent_tools)

                # Initialize external agent for non-manual input if needed
                logger1, client1, agent_tools1 = None, None, None
                if not MANUAL_INPUT:
                    logger1, client1, agent_tools1 = await initialize_agent()

                # Initialize queue and busy state
                input_queue = asyncio.Queue()
                self.is_busy = False

                # Run input collection and processing concurrently
                try:
                    await asyncio.gather(
                        self.collect_inputs(input_queue, logger1, client1, agent_tools1),
                        self.process_inputs(input_queue, agent, logger1, agent_tools1),
                        return_exceptions=True
                    )
                except asyncio.CancelledError:
                    self.logger.info("Tasks cancelled")
                except KeyboardInterrupt:
                    self.logger.info("Exiting via KeyboardInterrupt")

            except Exception as e:
                self.logger.error(f"Session error: {str(e)}")
                raise
            finally:
                self.logger.info("Cleaning up client session")
                await self.client.close()

if __name__ == "__main__":
    try:
        agent = BrowserAgent(history_maxlen=5)
        asyncio.run(agent.run())
    except Exception as e:
        logging.error(f"Application error: {str(e)}")
        raise