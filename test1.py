import os
import logging
from typing import Literal
from deepagents.backends import FilesystemBackend
from deepagents.backends.protocol import SandboxBackendProtocol
from deepagents.backends.sandbox import BaseSandbox
from tavily import TavilyClient
from langchain_openai import ChatOpenAI
from deepagents import create_deep_agent
from dotenv import load_dotenv
from deepagents.middleware import FilesystemMiddleware

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('log.txt', encoding='utf-8'),
        logging.StreamHandler()  # 同时输出到控制台
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

os.environ["AGENTRUN_ACCESS_KEY_ID"] = "your-access-key-id"
os.environ["AGENTRUN_ACCESS_KEY_SECRET"] = "your-access-key-secret"
os.environ["AGENTRUN_ACCOUNT_ID"] = "your-account-id"
os.environ["AGENTRUN_REGION"] = "cn-hangzhou"

def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search"""
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )

api_key = "21615d39960544f0fed894d9f07c5b8b.0AVks96E9RavRR66"
base_url = "https://open.bigmodel.cn/api/paas/v4"


model=ChatOpenAI(
        model="glm-5.1",
        api_key=api_key,
        base_url=base_url
)

research_subagent = {
    "name": "research-agent",
    "description": "Used to research more in depth questions",
    "system_prompt": "You are a great researcher",
    "tools": [internet_search],
    "model": model,  # Optional override, defaults to main agent model
}
subagents = [research_subagent]

agent = create_deep_agent(
    name="main-agent",
    model=model,
    system_prompt="You are a helpful research assistant",
    backend=FilesystemBackend(root_dir="./agent_files", virtual_mode=True),
    subagents=subagents
)

for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "你好，帮我把内容“121212”写到text.md中"}]},
    stream_mode=["messages", "updates"],
    subgraphs=True,
    version="v2",
):
    if chunk["type"] == "updates":
        if chunk["ns"]:
            # Subagent event - namespace identifies the source
            logger.info(f"[subagent: {chunk['ns']}]")
        else:
            # Main agent event
            logger.info("[main agent]")
        logger.info(chunk["data"])
    elif chunk["type"] == "messages":
        logger.info(chunk["data"])
        