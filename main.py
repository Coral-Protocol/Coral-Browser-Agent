import asyncio
import json
import logging
import os
import traceback
from contextlib import AsyncExitStack
from datetime import datetime
from logging.handlers import RotatingFileHandler
from urllib.parse import urlencode
from dotenv import load_dotenv
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_mcp_adapters.client import MultiServerMCPClient
from utils.browser_agent import browser_create_agent, process_agent_query, initialize_browser_session

class JsonFormatter(logging.Formatter):
    """Custom JSON formatter for logging."""
    def format(self, record):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno
        }
        return json.dumps(log_entry)

def setup_logging():
    """Configure logging with JSON formatting and rotating file handler."""
    log_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{timestamp}.log")

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    return logger

async def main():
    """Main entry point for the web agent application."""
    logger = setup_logging()

    # Load environment variables
    runtime = os.getenv("CORAL_ORCHESTRATION_RUNTIME")
    if runtime is None:
        load_dotenv()

    # Retrieve configuration
    base_url = os.getenv("CORAL_SSE_URL")
    agent_id = os.getenv("CORAL_AGENT_ID")

    if not all([base_url, agent_id]):
        logger.error("Missing required environment variables")
        raise ValueError("CORAL_SSE_URL and CORAL_AGENT_ID must be set")

    # Construct server URL
    coral_params = {
        "agentId": agent_id,
        "agentDescription": "Web agent for web browsing and surfing"
    }
    query_string = urlencode(coral_params)
    coral_server_url = f"{base_url}?{query_string}"
    logger.info(f"Connecting to Coral Server: {coral_server_url}")

    async with AsyncExitStack() as exit_stack:
        # Setup image directory
        current_dir = os.getcwd()
        images_dir = os.path.join(current_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

        # Initialize browser session and agent
        session, agent_tools_description = await initialize_browser_session(exit_stack, images_dir)
        agent_chain = await browser_create_agent(session)

        # Main agent loop
        tool_result = None
        last_tool_call = None
        step = 0

        while True:
            try:
                input_query = input("INPUT: ")
                await process_agent_query(
                    input_query,
                    tool_result,
                    last_tool_call,
                    step,
                    agent_chain,
                    agent_tools_description,
                    session
                )
                step += 1
            except Exception as e:
                logger.error(f"Error in agent loop: {str(e)}")
                logger.debug(f"Traceback: {traceback.format_exc()}")
                await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(main())