import urllib.parse
from dotenv import load_dotenv
import os, json, asyncio, traceback
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_tool_calling_agent, AgentExecutor
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
import json
from utils.browser_agent import browser_create_agent, load_browser_tools, execute_tool_call, process_agent_query
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

        # RUN IN NPX MODE     
        command = "npx"
        args = ["@playwright/mcp@latest", "--output-dir=/images"]


        # RUN BELOW ARGS WHENRUNNING IN SERVER (eg: WSL) IN HEADLESS MODE IN DOCKER
        # command = "docker"
        # args = [
        #     "run",
        #     "-i",
        #     "--rm",
        #     "--init",
        #     "--pull=always",
        #     "-v",
        #     f"{images_dir}:/images",
        #     "mcr.microsoft.com/playwright/mcp",
        #     "--no-sandbox",
        #     "--output-dir=/images",
        #     "--viewport-size=1920,1080"
        #  ]

        server_params = StdioServerParameters(
            command=command,
            args=args,
            env=None
        )

        stdio_transport = await exit_stack.enter_async_context(stdio_client(server_params))
        stdio, client = stdio_transport
        session = await exit_stack.enter_async_context(ClientSession(stdio, client))
        await session.initialize()

        response = await session.list_tools()
        agent_tools = response.tools
        # logger.info("Available Playwright MCP Tools:")
        # for tool in agent_tools:
        #     logger.info(f"- {tool.name}: {tool.description or 'No description available'}")
        
        tools = load_browser_tools()
        if not tools:
            logger.warning("No browser tools loaded, proceeding with empty toolset")
        
        # Format tools for inclusion in the prompt
        tools_description = ""
        for tool in tools:
            tools_description += (
                f"Tool: {tool['name']}\n"
                f"Description: {tool['description']}\n"
                f"Input Schema: {json.dumps(tool['inputSchema'], indent=2)}\n\n"
            )

        agent_tools_description = tools_description

        agent_chain = await browser_create_agent(agent_tools)
        await process_agent_query(session, agent_chain, agent_tools_description)
        

if __name__ == "__main__":
    asyncio.run(main())