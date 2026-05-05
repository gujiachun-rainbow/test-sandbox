## 项目环境

本项目使用 `uv` 管理虚拟环境（`.venv/`），所有操作前需先激活：

```bash
source .venv/Scripts/activate
```

- 安装依赖：`uv sync`
- 运行脚本：`python main.py`
- 环境变量通过 `.env` 文件配置，`main.py` 入口处用 `load_dotenv()` 自动加载

## Agent skills

### Issue tracker

Issues are tracked as GitHub issues in this repository. See `docs/agents/issue-tracker.md`.

### Triage labels

Uses default triage label vocabulary (needs-triage, needs-info, ready-for-agent, ready-for-human, wontfix). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context layout with CONTEXT.md and docs/adr/ at the repo root. See `docs/agents/domain.md`.
