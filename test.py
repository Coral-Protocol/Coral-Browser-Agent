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
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from typing import Dict, Any
from aiohttp import ClientSession

SLEEP_INTERVAL = 2 


class AgentArgs(BaseModel):
    input_query: str = Field(..., description="The user's input query")
    tool_result: Dict[str, Any] = Field(default_factory=dict, description="Result from the previous tool call")
    last_tool_call: str = Field(default="", description="JSON of the last tool call to avoid redundancy")
    step: int = Field(default=0, description="Current step number for logging")
    agent_chain: Any = Field(..., description="The LangChain chain for agent invocations")
    playwright_mcp_tools_description: str = Field(default="", description="Description of available tools")

    class Config:
        arbitrary_types_allowed = True  # Allow unsupported types like ClientSession

async def process_browsing_tools(args: AgentArgs, session: ClientSession) -> dict:
    """
    Process a query using the web browser agent.
    
    Args:
        args: AgentArgs object containing:
            input_query: The user's input query
            tool_result: Result from the previous tool call
            last_tool_call: JSON of the last tool call to avoid redundancy
            step: Current step number for logging
            agent_chain: The LangChain chain for agent invocations
            playwright_mcp_tools_description: Description of available tools
        session: The browser session
    
    Returns:
        Dictionary containing the result of the browser query
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Processing query at step {args.step}: {args.input_query}")
    await asyncio.sleep(2)  # Think for 2 seconds as per prompt
    try:
        if not args.agent_chain or not session:
            logger.error("Browser agent or session not initialized")
            return {"error": "Browser agent or session not initialized"}
        
        browser_result = await process_agent_query(
            input_query=args.input_query,
            tool_result=args.tool_result, 
            last_tool_call=args.last_tool_call,
            step=args.step,
            agent_chain=args.agent_chain,
            agent_tools_description=args.playwright_mcp_tools_description,
            session=session
        )
        
        await asyncio.sleep(3)  # Think for 3 seconds to evaluate response
        logger.info(f"Query processed successfully at step {args.step}")
        return {
            "status": "success",
            "result": browser_result,
            "step": args.step + 1
        }
    except Exception as e:
        logger.error(f"Failed to process browser query at step {args.step}: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to process browser query: {str(e)}"
        }

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
    """Generate a description of tools for the agent prompt."""
    return "\n".join(
        f"Tool: {tool.name}, Schema: {json.dumps(tool.args).replace('{', '{{').replace('}', '}}')}"
        for tool in tools
    )

async def create_agent(coral_tools, agent_tools):
    """Create and configure the agent with the given tools."""
    combined_tools = coral_tools + agent_tools
    
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are an agent interacting with tools from Coral Server and your own tools. Your task is to perform any instructions from other agents. 
            Follow these steps in order:
            1. Call wait_for_mentions from coral tools (timeoutMs: 20000) to receive mentions from other agents.
            2. When you receive a mention, keep the thread ID and the sender ID.
            3. Take 2 seconds to think about the content (instruction) of the message and check only the list of your tools available for action.
            4. Check the tool schema and make a plan in steps for the task you want to perform. You can call the tool `process_browsing_tools` with the following parameters:
               - input_query: The user's input query (string, required) derived from the mention's content.
               - tool_result: Result from the previous tool call (dictionary, default empty dict).
               - last_tool_call: JSON string of the last tool call to avoid redundancy (string, default empty string).
               - step: Current step number for logging (integer, default 0, must be non-negative).
               - agent_chain: The LangChain chain for agent invocations (required).
               - playwright_mcp_tools_description: Description of available tools (string, default empty string).
               - session: The browser session (ClientSession, required).
            5. Only call the tools you need to perform for each step of the plan to complete the instruction in the content.
            6. Take 3 seconds to evaluate the content and confirm you have executed the instruction to the best of your ability using the tools. Use the tool's output as the response content.
            7. Use `send_message` from coral tools to send a message in the same thread ID to the sender ID you received the mention from, with the tool's output as content.
            8. If any error occurs, use `send_message` to send a message in the same thread ID to the sender ID you received the mention from, with content: "error".
            9. Always respond back to the sender agent even if you have no answer or error.
            10. Wait for 2 seconds and repeat the process from step 1.

            These are the list of coral tools: {coral_tools_description}
            These are the list of your tools: {agent_tools_description}
            Additional parameters:
            - Tool result: {tool_result}
            - Last tool call: {last_tool_call}
            - Current step: {step}
            - Agent chain: {agent_chain}
            - Playwright MCP tools description: {playwright_mcp_tools_description}
            - Session: {session}

            """
        ),
        ("placeholder", "{agent_scratchpad}")
    ])

    model = init_chat_model(
        model=os.getenv("MODEL_NAME"),
        model_provider=os.getenv("MODEL_PROVIDER"),
        api_key=os.getenv("MODEL_API_KEY"),
        temperature=float(os.getenv("MODEL_TEMPERATURE", 0.0)),
        max_tokens=int(os.getenv("MODEL_MAX_TOKENS", 8000)),
        base_url=os.getenv("MODEL_BASE_URL", None)
    )
    agent = create_tool_calling_agent(model, combined_tools, prompt)
    return AgentExecutor(agent=agent, tools=combined_tools, verbose=True)

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

    # Initialize client
    timeout_ms = int(os.getenv("TIMEOUT_MS", 300))
    client = MultiServerMCPClient(
        connections={
            "coral": {
                "transport": "sse",
                "url": coral_server_url,
                "timeout": timeout_ms,
                "sse_read_timeout": timeout_ms,
            }
        }
    )
    coral_tools = await client.get_tools(server_name="coral")
    coral_tools_description = get_tools_description(coral_tools)

    async def process_browsing_tools_wrapper(args: AgentArgs) -> dict:
        async with ClientSession() as session:
            return await process_browsing_tools(args, session)

    agent_tools = [
        StructuredTool.from_function(
            name="process_browsing_tools",
            func=None,
            coroutine=process_browsing_tools_wrapper,
            description="Processes a query using a web browser agent to retrieve or interact with web content.",
            args_schema=AgentArgs
        )
    ]
    agent_tools_description = get_tools_description(agent_tools)

    print(coral_tools_description)
    print(agent_tools_description)

    agent_executor = await create_agent(coral_tools, agent_tools)


     # Initialize browser session and agent
    async with AsyncExitStack() as exit_stack:
        session, playwright_mcp_tools_description = await initialize_browser_session(exit_stack, images_dir)
        agent_chain = await browser_create_agent(session)

        tool_result = {}  # Initialize to match AgentArgs default
        last_tool_call = ""  # Initialize to match AgentArgs default
        step = 0
        while True:
            try:
                logger.info("Starting new agent invocation")
                # Fetch input_query from wait_for_mentions
 
                await agent_executor.ainvoke({
                    "agent_scratchpad": [],
                    "coral_tools_description": coral_tools_description,
                    "agent_tools_description": agent_tools_description,
                    "tool_result": tool_result,
                    "last_tool_call": last_tool_call,
                    "step": step,
                    "agent_chain": agent_chain,
                    "playwright_mcp_tools_description": playwright_mcp_tools_description,
                    "session": session,
                })

                step += 1

                await asyncio.sleep(SLEEP_INTERVAL)
            except Exception as e:
                logger.error(f"Error in agent loop: {str(e)}")
                logger.error(traceback.format_exc())

                await asyncio.sleep(SLEEP_INTERVAL)


if __name__ == "__main__":
    asyncio.run(main())