# FutureX-Past Benchmark Runner

这是一个精简的 MiroFlow 项目版本，专门用于运行 FutureX-Past 数据集的 benchmark 评估。

## 项目结构

```
futureX_miro/
├── run_futurex_past_fast.py          # 主入口文件（快速版）
├── common_benchmark_past.py          # Benchmark 核心逻辑
├── requirements.txt                   # Python 依赖
├── config/                            # 配置文件
│   ├── agent_futurex_past_fast.yaml  # Agent 配置
│   ├── benchmark/                     # Benchmark 配置
│   │   └── futurex_past.yaml
│   ├── tool/                          # 工具配置
│   │   ├── tool-searching-firecrawl-exa.yaml
│   │   └── tool-reading-firecrawl-exa.yaml
│   └── agent_prompts/                 # Agent Prompt 模板
│       ├── base_agent_prompt.py
│       └── main_boxed_answer.py
├── src/                               # 源代码
│   ├── core/                          # 核心组件
│   │   ├── pipeline.py               # 任务执行管道
│   │   └── orchestrator.py           # 任务编排器
│   ├── tool/                          # 工具管理
│   │   ├── manager.py                # 工具管理器
│   │   └── mcp_servers/              # MCP 服务器实现
│   │       ├── searching_mcp_server_firecrawl_exa.py
│   │       ├── reading_mcp_server_firecrawl_exa.py
│   │       └── browser_session.py
│   ├── llm/                           # LLM 客户端
│   │   ├── client.py                 # LLM 客户端工厂
│   │   ├── provider_client_base.py   # 客户端基类
│   │   └── providers/                # 各种 LLM Provider
│   │       ├── gpt5_openai_client.py
│   │       └── claude_openrouter_client.py
│   ├── logging/                       # 日志系统
│   │   ├── logger.py                 # 日志配置
│   │   └── task_tracer.py            # 任务追踪
│   └── utils/                         # 工具函数
│       ├── io_utils.py               # IO 工具
│       ├── tool_utils.py             # 工具函数
│       └── summary_utils.py          # 摘要生成
└── utils/                             # 评估工具
    └── eval_utils.py                 # 答案验证
```

## 核心依赖关系

### 主要模块
1. **入口模块**: `run_futurex_past_fast.py` → `common_benchmark_past.py`
2. **Pipeline**: `src.core.pipeline` → 创建并执行任务管道
3. **Orchestrator**: `src.core.orchestrator` → 协调 Agent 和工具
4. **Tool Manager**: `src.tool.manager` → 管理 MCP 工具调用
5. **LLM Client**: `src.llm.client` → 统一 LLM 接口

### 数据流
```
用户输入 → Pipeline → Orchestrator → LLM Client ↔ Tool Manager ↔ MCP Servers
                                            ↓
                                      Task Tracer (日志)
                                            ↓
                                       评估结果输出
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

复制示例文件并编辑：

```bash
cp .env.example .env
nano .env  # 或使用其他编辑器
```

配置以下必需的 API 密钥：

```bash
# OpenRouter API (用于 GPT-5)
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# 搜索和网页抓取 API
FIRECRAWL_API_KEY=your_firecrawl_api_key
EXA_API_KEY=your_exa_api_key

# Hugging Face Token (下载数据集用)
HF_TOKEN=your_huggingface_token

# 数据目录
DATA_DIR=data
```

获取 API 密钥：
- OpenRouter: https://openrouter.ai/keys
- Firecrawl: https://firecrawl.dev
- Exa: https://exa.ai
- Hugging Face: https://huggingface.co/settings/tokens

### 3. 下载数据集

```bash
python prepare_futurex.py
```

这将从 Hugging Face 下载 FutureX-Past 数据集并保存到 `data/futurex/standardized_data_past.jsonl`。

### 4. 运行评估

#### 测试运行（1个任务）

```bash
python run_futurex_past_fast.py \
    benchmark.execution.max_tasks=1 \
    output_dir="logs/test"
```

#### 完整评估（所有任务）

```bash
python run_futurex_past_fast.py \
    benchmark.execution.max_tasks=851 \
    benchmark.execution.max_concurrent=10 \
    output_dir="logs/full_run"
```

#### 其他配置示例

```bash
# 指定输出目录
python run_futurex_past_fast.py output_dir="logs/my_run"

# 限制任务数量
python run_futurex_past_fast.py benchmark.execution.max_tasks=10

# 调整并发数
python run_futurex_past_fast.py benchmark.execution.max_concurrent=5
```

## 配置说明

### Agent 配置 (`config/agent_futurex_past_fast.yaml`)

- **LLM 配置**: 使用 GPT-5 通过 OpenRouter
- **工具配置**: Firecrawl + Exa 搜索和阅读
- **优化参数**:
  - `max_turns`: 6（限制最大轮次）
  - `max_tool_calls_per_turn`: 6（每轮最多工具调用）
  - `max_tokens`: 16000（输出限制）
  - `reasoning_effort`: low（推理速度优化）

### Benchmark 配置 (`config/benchmark/futurex_past.yaml`)

- **数据集**: FutureX-Past
- **评估**: pass@1（单次尝试）
- **并发**: 可配置（默认 8）

## 主要功能

### 1. 任务加载与执行
- 从 JSONL 文件加载任务
- 支持任务白名单过滤
- 并行执行多个任务
- 自动缓存已完成的结果

### 2. 工具系统
- **搜索工具**: Exa AI 搜索 + Firecrawl 网页抓取
- **阅读工具**: 支持多种文件格式（PDF、Excel、图片等）
- **时间约束**: 自动应用历史时间过滤（硬约束）

### 3. LLM 集成
- 支持 GPT-5（通过 OpenRouter）
- 支持 Claude（通过 Anthropic/OpenRouter）
- 统一的工具调用接口
- 自动重试和错误处理

### 4. 日志与追踪
- 完整的任务执行日志
- 工具调用记录
- LLM 对话历史
- 评估结果保存

## 输出文件

运行后会在输出目录生成：

```
logs/full_run/
├── task_{task_id}_attempt_1.json    # 每个任务的详细日志
├── benchmark_results.jsonl          # 所有任务的结果
└── .hydra/                          # Hydra 配置快照
    ├── config.yaml
    └── overrides.yaml
```

## 评估结果

每个任务的结果包括：
- `task_id`: 任务 ID
- `model_response`: 模型完整响应
- `model_boxed_answer`: 提取的最终答案（`\boxed{...}` 格式）
- `judge_result`: 评估结果（CORRECT/INCORRECT/SKIPPED）
- `status`: 执行状态（completed/failed）

## 故障排查

### 常见问题

1. **API Key 错误**
   - 检查 `.env` 文件中的 API key 是否正确
   - 确保 OpenRouter 账户有足够的余额

2. **数据集路径错误**
   - 检查 `DATA_DIR` 环境变量
   - 确保 `data/futurex/standardized_data_past.jsonl` 存在

3. **工具调用失败**
   - 检查 Firecrawl 和 Exa API key
   - 查看任务日志中的错误信息

4. **内存不足**
   - 减少 `max_concurrent` 参数
   - 限制 `max_tasks` 数量

## 主要文件说明

- `prepare_futurex.py` - 下载 FutureX-Past 数据集
- `run_futurex_past_fast.py` - 运行评估的主入口
- `common_benchmark_past.py` - Benchmark 核心逻辑
- `config/agent_futurex_past_fast.yaml` - Agent 配置文件
- `requirements.txt` - Python 依赖列表

## 与原始 MiroFlow 的区别

这个版本是精简版，专注于 FutureX-Past benchmark：

- **移除**: UI、交互式运行、多余的工具
- **保留**: 核心 Pipeline、Orchestrator、LLM 客户端
- **优化**: 更快的推理速度、更少的资源消耗

## License

Apache-2.0 License

## 致谢

基于 [MiroFlow](https://github.com/MiroMindAI/MiroFlow) 项目开发。

