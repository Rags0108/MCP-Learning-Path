# utils.py
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from prompt import user_goal_prompt
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_community.chat_models import ChatLlamaCpp
from typing import Optional, Callable, Any
import asyncio
import logging

logging.basicConfig(level=logging.INFO)
cfg = RunnableConfig(recursion_limit=50)

def initialize_model():
    return ChatLlamaCpp(
        model_path=r"D:\Project\llm\model\grok-gamma\Grok-3-reasoning-gemma3-12B-distilled-HF.Q3_K_S.gguf",
        n_ctx=4096,  # Reduced for compatibility
        max_tokens=2048,
        temperature=0.7,
        n_batch=16,  # Adjust for available RAM
        verbose=True
    )

def truncate_text(text, max_words=1800):
    words = text.split()
    if len(words) > max_words:
        return " ".join(words[:max_words])
    return text

async def setup_agent_with_tools(
    youtube_pipedream_url: str,
    drive_pipedream_url: Optional[str] = None,
    notion_pipedream_url: Optional[str] = None,
    progress_callback: Optional[Callable[[str], None]] = None
) -> Any:
    try:
        if progress_callback:
            progress_callback("Setting up agent with tools... ✅")

        tools_config = {
            "youtube": {
                "url": youtube_pipedream_url,
                "transport": "streamable_http"
            }
        }
        if drive_pipedream_url:
            tools_config["drive"] = {
                "url": drive_pipedream_url,
                "transport": "streamable_http"
            }
            if progress_callback:
                progress_callback("Added Google Drive integration... ✅")
        if notion_pipedream_url:
            tools_config["notion"] = {
                "url": notion_pipedream_url,
                "transport": "streamable_http"
            }
            if progress_callback:
                progress_callback("Added Notion integration... ✅")

        if progress_callback:
            progress_callback("Initializing MCP client... ✅")

        mcp_client = MultiServerMCPClient(tools_config)

        if progress_callback:
            progress_callback("Getting available tools... ✅")

        tools = await mcp_client.get_tools()

        if progress_callback:
            progress_callback("Creating AI agent... ✅")

        model = initialize_model()
        agent = create_react_agent(model, tools)

        if progress_callback:
            progress_callback("Setup complete! Starting to generate learning path... ✅")

        return agent

    except Exception as e:
        logging.error(f"Error in setup_agent_with_tools: {str(e)}")
        raise

def run_agent_sync(
    youtube_pipedream_url: str,
    drive_pipedream_url: Optional[str] = None,
    notion_pipedream_url: Optional[str] = None,
    user_goal: str = "",
    progress_callback: Optional[Callable[[str], None]] = None
) -> dict:
    async def _run():
        try:
            agent = await setup_agent_with_tools(
                youtube_pipedream_url=youtube_pipedream_url,
                drive_pipedream_url=drive_pipedream_url,
                notion_pipedream_url=notion_pipedream_url,
                progress_callback=progress_callback
            )
            learning_path_prompt = (
                f"User Goal: {user_goal}\n"
                "You are an expert learning path generator. "
                "Provide a detailed, day-wise learning path for the user goal above. "
                "Do not output any actions or thoughts, only the final answer. "
                "Start your answer with 'Final Answer:'.\n"
                + user_goal_prompt
            )
            learning_path_prompt = truncate_text(learning_path_prompt)
            if progress_callback:
                progress_callback("Generating your learning path...")

            result = await agent.ainvoke(
                {"messages": [HumanMessage(content=learning_path_prompt)]},
                config=cfg
            )

            if progress_callback:
                progress_callback("Learning path generation complete!")

            return result
        except Exception as e:
            logging.error(f"Error in _run: {str(e)}")
            raise

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(_run())
    finally:
        loop.close()

