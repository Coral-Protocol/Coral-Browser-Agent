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
from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate
from utils.browser_agent import browser_create_agent, process_agent_query, initialize_browser_session
from utils.coral_config import load_config, get_tools_description, parse_mentions_response, mcp_resources_details



REQUEST_QUESTION_TOOL = "request-question"
ANSWER_QUESTION_TOOL = "answer-question"
WAIT_FOR_MENTIONS_TOOL = "wait-for-mentions"
MAX_CHAT_HISTORY = 3
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 8000
SLEEP_INTERVAL = 1
ERROR_RETRY_INTERVAL = 5

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

def get_tools_description(tools):
    return "\n".join(
        f"Tool: {tool.name}, Schema: {json.dumps(tool.args).replace('{', '{{').replace('}', '}}')}"
        for tool in tools
    )

async def main():
    """Main entry point for the web agent application."""
    logger = setup_logging()
    current_dir = os.getcwd()
    images_dir = os.path.join(current_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

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

    timeout = os.getenv("TIMEOUT_MS", 300)
    client = MultiServerMCPClient(
        connections={
            "coral": {
                "transport": "sse",
                "url": coral_server_url,
                "timeout": timeout,
                "sse_read_timeout": timeout,
            }
        }
    )
    coral_tools = await client.get_tools(server_name="coral")

    agent_tools = {tool.name: tool for tool in coral_tools}

    # Initialize browser session and agent
    async with AsyncExitStack() as exit_stack:
        session, agent_tools_description = await initialize_browser_session(exit_stack, images_dir)
        agent_chain = await browser_create_agent(session)

        # Main agent loop
        tool_result = None
        last_tool_call = None
        step = 0

        while True:
            try:
                logger.info("***********************Waiting for Mentions***********************")
                resources = await client.get_resources(server_name="coral")
                coral_resources = mcp_resources_details(resources)
                mentions_response = await agent_tools['wait_for_mentions'].ainvoke({
                    "timeoutMs": 30000
                })
                logger.info(f"Received mentions response: {mentions_response}")

                if isinstance(mentions_response, str) and "No new messages" in mentions_response:
                    await asyncio.sleep(SLEEP_INTERVAL)
                    continue
                    
                messages = parse_mentions_response(mentions_response)
                if not messages or not messages[0].get('threadId'):
                    await asyncio.sleep(SLEEP_INTERVAL)
                    continue
                    
                message = messages[0]
                thread_id = message.get('threadId')
                sender_id = message.get('senderId')
                content = message.get('content')

                if not all([thread_id, sender_id, content]):
                    logger.warning(f"Missing message fields: thread_id={thread_id}, sender_id={sender_id}")
                    await agent_tools['send_message'].ainvoke({
                        "threadId": thread_id,
                        "content": "Error: Missing message fields",
                        "mentions": [sender_id]
                    })
                    await asyncio.sleep(SLEEP_INTERVAL) 
                    continue

                input_query = "go to google"

                # Process the query with the web browser agent
                browser_result = await process_agent_query(
                    input_query,
                    tool_result,
                    last_tool_call,
                    step,
                    agent_chain,
                    agent_tools_description,
                    session
                )
                step += 1

                answer = browser_result
                logger.info(f"Final answer: {answer}")
                
                await agent_tools['send_message'].ainvoke({
                    "threadId": thread_id,
                    "content": answer,
                    "mentions": [sender_id]
                })
                logger.info(f"Sent response to thread_id={thread_id}, sender_id={sender_id}, content: {answer}")

                await asyncio.sleep(SLEEP_INTERVAL)




            except Exception as e:
                logger.error(f"Error in agent loop: {str(e)}")
                logger.debug(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    asyncio.run(main())