from deepagents.backends import FilesystemBackend
from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI
import os

# 1. 初始化：限定Agent只能操作 /agent_files 目录
fs_backend = FilesystemBackend(
    root_dir="./agent_files",  # 沙箱根目录
    virtual_mode=True,         # 隐藏真实路径，对外显示为 /
)

# 2. 配置 API 密钥和基础 URL
# 建议从环境变量或配置文件中读取，这里为示例直接设置
api_key = "21615d39960544f0fed894d9f07c5b8b.0AVks96E9RavRR66"
base_url = "https://open.bigmodel.cn/api/paas/v4"

# 3. 创建Agent
agent = create_deep_agent(
    model=ChatOpenAI(
        model="glm-5.1",
        api_key=api_key,
        base_url=base_url
    ),
    backend=lambda rt: fs_backend
)

# 3. 使用：Agent 读写 /notes.txt → 实际写入 /agent_files/notes.txt
response = agent.invoke({
    "messages": [{"role": "user", "content": "写入笔记：今天学习LangChain后端;不需要写其他内容。"}]
})

print(response)