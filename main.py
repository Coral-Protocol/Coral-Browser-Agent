import urllib.parse
from dotenv import load_dotenv
import os, json, asyncio, traceback
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_tool_calling_agent, AgentExecutor
from contextlib import AsyncExitStack
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
import json
from utils.browser_agent import browser_create_agent, process_agent_query, initialize_browser_session
import logging
from logging.handlers import RotatingFileHandler

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno
        }
        return json.dumps(log_entry)

log_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"{timestamp}.log")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
handler.setFormatter(JsonFormatter())
logger.addHandler(handler)

async def main():
    runtime = os.getenv("CORAL_ORCHESTRATION_RUNTIME", None)
    if runtime is None:
        load_dotenv()

    base_url = os.getenv("CORAL_SSE_URL")
    agentID = os.getenv("CORAL_AGENT_ID")

    coral_params = {
        "agentId": agentID,
        "agentDescription": "Web agent for web browsing and surfing"
    }

    query_string = urllib.parse.urlencode(coral_params)
    CORAL_SERVER_URL = f"{base_url}?{query_string}"
    logger.info(f"Connecting to Coral Server: {CORAL_SERVER_URL}")

    async with AsyncExitStack() as exit_stack:
        current_dir = os.getcwd()
        images_dir = os.path.join(current_dir, "images")
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)

        session, agent_tools_description = await initialize_browser_session(exit_stack, images_dir)
        agent_chain = await browser_create_agent(session)

        tool_result = None 
        last_tool_call = None
        step = 0
        while True:
            try:
                input_query = input("INPUT: ")
                await process_agent_query(input_query, tool_result, last_tool_call, step, agent_chain, agent_tools_description, session)
                
            except Exception as e:
                logger.error(f"Error in agent loop: {e}")
                logger.debug(f"Traceback: {traceback.format_exc()}")
                await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(main())