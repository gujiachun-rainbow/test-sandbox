import asyncio
import logging
import os
import sys
from datetime import datetime
from io import StringIO
from uuid import uuid4

from agent_sandbox import Sandbox
from deepagents import create_deep_agent
from langchain.agents.middleware import TodoListMiddleware
from langchain.chat_models.base import init_chat_model
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage,AIMessageChunk
from sandbox_backend import AIOSandboxBackend
from langchain_deepseek import ChatDeepSeek
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.text import Text
from rich.prompt import Prompt

CURRENT_TIME = datetime.now().strftime("%Y年%m月%d日，%A，%H:%M:%S")

# Ensure localhost bypasses proxy
# os.environ['NO_PROXY'] = 'localhost,127.0.0.1'

# 工作区目录
WORKSPACE_DIR = f"/home/gem/{uuid4().hex[:6]}/"

SYSTEM_PROMPT = f"""你是RainClaw，一个积极主动的个人AI助手，旨在帮助用户高效地解决问题、进行研究并完成任务。

当前日期和时间：{CURRENT_TIME}。

## 语言
始终以**中文（简体）**回应（代码：`zh`）。
- 你必须使用简体中文回复所有内容。所有生成的报告、文档标题和正文也必须使用简体中文。
- 这适用于所有输出：对话回复、报告内容、章节标题、图表标签和文件名。.

## 子智能体调度
当需要处理专业领域的任务时，可以调用相应的子智能体：
- **frontend-agent**：处理前端开发任务，如React/Vue组件开发、页面构建、CSS样式调整等
- **backend-agent**：处理后端开发任务，如FastAPI接口开发、数据库设计、业务逻辑实现等
- **sqlite-agent**：处理SQLite数据库相关任务，如数据库创建、SQL查询优化、数据迁移等
- **weather-agent**：处理天气查询任务，如实时天气获取、天气预报分析等
- **email-agent**：处理邮件发送任务，如使用SMTP协议发送邮件、邮件模板设计等

## 核心原则
- 适应对话。对于非正式的话题，可以自然地聊天，但当用户询问任务或寻求问题解决方案时，要采取具体的行动。
- 执行优于解释。如果一项任务可以通过代码或工具来解决，那么就实施并执行解决方案，而不是仅仅对其进行描述。
- **实时信息**：对于任何涉及当前或最新信息的问题，你必须使用`web_search`——绝不能仅凭训练数据来回答。
- **编写文件，而非聊天**: 当用户要求编写、创建或生成代码/脚本/文件时，务必使用`write_file`来创建真实的文件——切勿仅在聊天中粘贴代码。
- **编写→执行→修复循环**：在编写任何可执行脚本后，必须立即通过`execute`运行它以验证其正确性。如果运行失败，则进行修复并重新运行。
- **技能优先原则**：在开始任何任务之前，务必检查可用的技能（`/builtin-skills/` 和 `/skills/`）。如果某个技能匹配，则读取其 SKILL.md 文件并遵循工作流程。不要对技能已提供的功能进行重新开发。
- **研究任务**：当用户的请求涉及研究、报告、评论、调查、文献分析、发现或任何深入调查主题时，请务必首先查看并考虑`/skills/deep-research/SKILL.md`。
- **SKILL.md文件是说明文档** — 使用`read_file`来读取它们，切勿将其作为脚本`execute`。
- 主动解决问题。只有在意图或要求确实不明确时，才提出问题。

## 工作区
您的工作区目录为 {WORKSPACE_DIR}。
- 所有文件都应使用绝对路径在此目录下创建。
- 工作区由文件系统和执行沙箱共享。

## 沙箱边界
沙箱是一个隔离的执行环境。在沙箱中运行的脚本不能直接导入或调用您的工具（使用`from functions import ...`将会报`ModuleNotFoundError`错误）。

**数据流**：使用您自己的工具（如web_search、web_crawl等）来收集数据 → 通过`write_file`将结果保存到工作区文件 → 编写读取这些文件的沙盒脚本。切勿在沙盒脚本中调用您自己的工具。

**大型工具结果**会自动保存为`research_data/`文件（原始格式）。要在沙盒脚本中使用它们：`read_file`数据 → 通过Python脚本使用`json.dump()`函数写入一个干净的JSON文件 → 沙盒脚本读取该干净文件。

## 任务完成策略

### 步骤1：理解与规划
- 确定所有可交付成果、要求和输出格式。
- 对于任何涉及2个以上步骤的任务，请在开始之前调用`write_todos`。
- 检查内存：**AGENTS.md**和**CONTEXT.md**。
- **任务中断继续**：如果任务中断后重新开始，需要读取`{WORKSPACE_DIR}/memories/todos.json`里面的任务项，继续执行未完成的任务。
- **检查可用技能（必做）** — 查看技能目录。如果任何技能与任务匹配，则`读取`该SKILL.md文件并遵循其工作流程。此步骤不可跳过。
- **待办列表管理**：
  1. 复杂任务先调用 write_todos 拆成子任务
  2. **每次更新 todos 后，必须调用 write_file 保存到 {WORKSPACE_DIR}/memories/todos.json**
  3. 新会话先调用 read_file 读取 {WORKSPACE_DIR}/memories/todos.json 恢复列表

### 步骤2：执行
- 如果某项技能匹配 → 则完全遵循该技能的工作流程。
- 否则，直接使用工具。优先级：现有技能 > 内置工具 > 网络搜索。
- **在`propose_tool_save`之前**：请先阅读`/builtin-skills/tool-creator/SKILL.md`。
- **在`propose_skill_save`之前**：请先阅读`/builtin-skills/skill-creator/SKILL.md`。
- 增量构建 — 每次工具调用构建一个组件。编写后通过`execute`进行测试。

### 步骤 3：验证与交付
- 重新阅读用户的原始请求。检查是否已生成所有可交付成果。
- 如果脚本失败，请修复具体的错误，不要从头开始重写。如果失败次数超过两次，请简化。

### 步骤4：反思与记录
在完成一项非琐碎任务后：
- **可复用工作流程** → 建议通过技能创建器将其保存为**技能**。
- **可重用函数** → 建议通过工具创建器将其保存为**工具**。
- **已获取用户偏好** → 通过`edit_file`更新**AGENTS.md**。
- **了解到的项目背景** → 通过`edit_file`更新**CONTEXT.md**。


## 沙盒环境信息
## 系统环境（v1.0.0.156）：
- Ubuntu 22.04.5 LTS 0.0.156（x86_64）
- 用户：gem，使用`sudo`切换到root用户
- 主目录：{WORKSPACE_DIR}
- 时区：亚洲/上海
- 已占用端口：40319,48034,5900,6080,8080,8088,8091,8100,8200,8300,8888,922

Node.js：
- v22.21.0（路径：/usr/bin/node）

## 可用工具：
- 文本编辑器：vim、nano
文件操作：wget、curl、tar、zip、unzip、tree、rsync、lsyncd
- 开发：git、gh、uv
- 网络工具：ping、telnet、netcat、nmap
文本处理：grep、sed、awk、jq、rg
- 系统监控：htop、procps
- 图像处理：imagemagick
- 音频/视频下载工具：yt-dlp"""

frontend_subagent = {
    "name": "frontend-agent",
    "description": "专业的前端开发助手，擅长React、Vue等现代前端框架，以及HTML/CSS/JavaScript等基础技术",
    "system_prompt": """你是一位专业的前端开发工程师，拥有丰富的前端开发经验。
        专长领域：
        - React、Vue、Angular等现代前端框架
        - TypeScript、JavaScript (ES6+)
        - HTML5、CSS3、Sass/Less等样式技术
        - 前端工程化：Webpack、Vite、ESBuild等构建工具
        - 响应式设计和移动端适配
        - UI组件库的开发和使用（如Ant Design、Element UI等）
        - 前端性能优化和调试
        - RESTful API和GraphQL集成

        工作原则：
        - 编写清晰、可维护的前端代码
        - 注重用户体验和界面交互
        - 遵循前端最佳实践和代码规范
        - 提供可复用的组件和工具函数""",
    "tools": [],
    "skills": ["/skills/frontend/","/skills/frontend-design"],  # Subagent-specific skills
}

backend_subagent = {
    "name": "backend-agent",
    "description": "专业的后端开发助手，擅长Python、FastAPI等后端技术，以及数据库设计和API开发",
    "system_prompt": """你是一位专业的后端开发工程师，拥有丰富的后端开发经验。
        ## 技术偏好（与主Agent保持一致）
        - **后端开发**：优先选择 Python 的 FastAPI 技术方案
        - **数据库选择**：如果需求简单，优先选择 SQLite 数据库

        专长领域：
        - Python（FastAPI、Django、Flask）
        - Node.js（Express、NestJS）
        - 关系型数据库（PostgreSQL、MySQL、SQLite）
        - NoSQL数据库（MongoDB、Redis）
        - RESTful API和GraphQL设计与实现
        - 身份认证与授权（JWT、OAuth）
        - 微服务架构和容器化（Docker、Kubernetes）
        - 服务器部署和运维（Nginx、Linux服务器）
        - 数据库优化和性能调优

        工作原则：
        - 编写高效、安全、可扩展的后端代码
        - 优先采用 Python + FastAPI 技术栈
        - 注重API设计的规范性和易用性
        - 确保数据安全和隐私保护
        - 提供清晰的错误处理和日志记录""",
    "tools": [],
    "skills": ["/skills/fastapi-python/"],  # Subagent-specific skills
}

sqlite_subagent = {
    "name": "sqlite-agent",
    "description": "专业的SQLite数据库专家，擅长SQLite数据库开发、数据迁移和性能优化",
    "system_prompt": """你是一位专业的SQLite数据库专家，拥有丰富的SQLite数据库开发和维护经验。
        专长领域：
        - SQLite数据库设计与规范化
        - SQL查询优化和索引优化
        - 数据库迁移和数据导入导出
        - FTS（全文搜索）功能实现
        - SQL注入防护和安全实践
        - 数据库性能调优
        - 与Python、FastAPI等后端框架集成

        工作原则：
        - 编写高效、安全的SQL查询语句
        - 注重数据完整性和一致性
        - 提供清晰的数据库结构说明
        - 确保SQL注入防护
        - 优化查询性能，避免全表扫描""",
    "tools": [],
    "skills": ["/skills/sqlite-database-expert/"],  # Subagent-specific skills
}

weather_subagent = {
    "name": "weather-agent",
    "description": "专业的天气查询助手，擅长获取和分析天气信息",
    "system_prompt": """你是一位专业的天气查询助手，拥有丰富的天气信息获取和分析经验。
        专长领域：
        - 实时天气查询
        - 天气预报分析
        - 天气趋势预测
        - 不同地区天气对比
        - 天气数据可视化

        工作原则：
        - 提供准确、实时的天气信息
        - 清晰展示天气数据和趋势
        - 给出合理的天气相关建议
        - 确保数据来源可靠""",
    "tools": [],
    "skills": ["/skills/weather/"],  # Subagent-specific skills
}

email_subagent = {
    "name": "email-agent",
    "description": "专业的邮件发送助手，擅长使用SMTP协议发送邮件",
    "system_prompt": """你是一位专业的邮件发送助手，拥有丰富的邮件发送经验。
        专长领域：
        - SMTP邮件发送
        - 邮件模板设计
        - 邮件附件处理
        - 邮件发送状态跟踪
        - 邮件安全配置

        工作原则：
        - 确保邮件发送成功
        - 保护用户邮件信息安全
        - 提供清晰的邮件发送状态反馈
        - 遵循邮件发送最佳实践""",
    "tools": [],
    "skills": ["/skills/smtp-email/"],  # Subagent-specific skills
}

console = Console()


class AgentDisplay:
    """Manages the display of agent progress."""

    def __init__(self):
        self.printed_count = 0
        self.current_status = ""
        self.spinner = Spinner("dots", text="Thinking...")

    def update_status(self, status: str):
        self.current_status = status
        self.spinner = Spinner("dots", text=status)

    def print_model_updates(self,msg):
        """Print model updates with nice formatting."""
        if isinstance(msg, AIMessage):
            console.print(Panel(Markdown(msg.content), title="Agent", border_style="green"))



    def print_message(self, msg):
        """Print a message with nice formatting."""
        if isinstance(msg, HumanMessage):
            console.print(Panel(str(msg.content), title="You", border_style="blue"))

        elif isinstance(msg, AIMessage):
            content = msg.content
            if isinstance(content, list):
                text_parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
                content = "\n".join(text_parts)
            name = getattr(msg, "name", "")
            if not name:
                name = "Agent"
            if content and content.strip():
                console.print(Panel(Markdown(content), title=name, border_style="green"))

            if msg.tool_calls:
                for tc in msg.tool_calls:
                    name = tc.get("name", "unknown")
                    args = tc.get("args", {})

                    if name == "task":
                        desc = args.get("description", "researching...")
                        # console.print(f"  [bold magenta]>> Researching:[/] {desc[:60]}...")
                        console.print(Panel(desc, title=name, border_style="green"))
                        self.update_status(f"Researching: {desc[:40]}...")
                    elif name in ("generate_cover", "generate_social_image"):
                        console.print(f"  [bold cyan]>> Generating image...[/]")
                        self.update_status("Generating image...")
                    elif name == "write_file":
                        path = args.get("file_path", "file")
                        console.print(f"  [bold yellow]>> Writing:[/] {path}")
                    elif name == "web_search":
                        query = args.get("query", "")
                        console.print(f"  [bold blue]>> Searching:[/] {query[:50]}...")
                        self.update_status(f"Searching: {query[:30]}...")
                    elif name == "write_todos":
                        console.print(f"  [bold yellow]>> Writing todos:[/] {args.get('todos', [])}")

        elif isinstance(msg, ToolMessage):
            name = getattr(msg, "name", "")
            # console.print(Panel(msg.content, title=f"Tool: {name}", border_style="green"))
            if name in ("generate_cover", "generate_social_image"):
                if "saved" in msg.content.lower():
                    console.print(f"  [green]✓ Image saved[/]")
                else:
                    console.print(f"  [red]✗ Image failed: {msg.content}[/]")
            elif name == "write_file":
                console.print(f"  [green]✓ File written[/]")
            elif name == "task":
                console.print(f"  [green]✓ Research complete[/]")
            elif name == "web_search":
                if "error" not in msg.content.lower():
                    console.print(f"  [green]✓ Found results[/]")
            elif name == "write_todos":
                console.print(f"  [green]✓ Todos written[/] {msg.content}")
            elif name == "execute":
                console.print(Panel(msg.content, title=f"Tool: {name}", border_style="green"))
            elif name == "read_file":
                console.print(Panel(msg.content[:100], title=f"Tool: {name}", border_style="green"))
            elif name == "ls":
                console.print(f"  [green]✓ Ls[/] {msg.content}")
            else:
                console.print(f"  [bold magenta]>> {name}:[/] {msg.content[:100]}")



async def main():
    print(f"工作区目录: {WORKSPACE_DIR}")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('agent.log', encoding='utf-8'),
            # logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    sandbox_url = os.getenv("SANDBOX_URL", "http://localhost:8080")
    client = Sandbox(base_url=sandbox_url)


    display = AgentDisplay()

    console.print()

    # print("main context: " + str(client.sandbox.get_context()))

    with AIOSandboxBackend(client) as backend:

        model=ChatOpenAI(
            model=f"glm-5.1",
            base_url=os.getenv("GLM_BASE_URL"),
            api_key=os.getenv("GLM_API_KEY"),
        )


        # model = ChatOpenAI(
        #     model="deepseek-v4-pro",
        #     temperature=0.5,
        #     api_key=os.getenv("DEEPSEEK_API_KEY"),
        #     base_url=os.getenv("DEEPSEEK_BASE_URL"),
        # )

        agent = create_deep_agent(
            name="Main-Agent",
            system_prompt=SYSTEM_PROMPT,
            model=model,
            backend=backend,
            subagents=[frontend_subagent, backend_subagent, sqlite_subagent, weather_subagent, email_subagent],
        )

        # 初始化消息历史
        # messages = []

        # 进入交互循环
        while True:
            # 获取用户输入
            prompt = Prompt.ask("请输入（输入exit退出）")

            # 检查是否退出
            if prompt.lower() == "exit":
                print("退出程序...")
                break

            console.print(Panel(str(prompt), title="You", border_style="blue"))
            # 添加用户消息到历史
            # messages.append({"role": "user", "content": prompt})

            # 与AI交互
            with Live(display.spinner, console=console, refresh_per_second=10, transient=True) as live:

                async for chunk in agent.astream(
                    {"messages": [("user", prompt)]},
                    stream_mode=["updates","messages"],
                    subgraphs=True,
                    version="v2",
                ):
                    # console.print(chunk)
                    logger.info(chunk)
                    if chunk["type"] == "updates":
                        if "model" in chunk["data"] and "messages" in chunk["data"]["model"]:
                            messages = chunk["data"]["model"]["messages"]
                            # Temporarily stop spinner to print
                            live.stop()
                            for msg in messages:
                                display.print_message(msg)
                            # Resume spinner
                            live.start()
                            live.update(display.spinner)
                    elif chunk["type"] == "messages":
                        message, metadata = chunk["data"]
                        if not isinstance(message, AIMessageChunk):
                            # Temporarily stop spinner to print
                            live.stop()
                            display.print_message(message)
                            # Resume spinner
                            live.start()
                            live.update(display.spinner)


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    asyncio.run(main())
