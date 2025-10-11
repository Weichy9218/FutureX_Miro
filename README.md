# FutureX-Past Benchmark Runner

基于 [MiroFlow](https://github.com/MiroMindAI/MiroFlow) 的 FutureX-Past 数据集评估工具。

## 快速复现

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API 密钥

```bash
cp .env.example .env
nano .env
```

必需配置：
```bash
OPENROUTER_API_KEY=your_key    # GPT-5 访问
FIRECRAWL_API_KEY=your_key     # 网页抓取
EXA_API_KEY=your_key            # 搜索引擎
HF_TOKEN=your_token             # 下载数据集
DATA_DIR=data
```

### 3. 下载数据集

```bash
python prepare_futurex.py
```

### 4. 运行评估

```bash
# 测试（1个任务）
python run_futurex_past_fast.py benchmark.execution.max_tasks=1

# 完整评估（851个任务）
python run_futurex_past_fast.py \
    benchmark.execution.max_tasks=851 \
    benchmark.execution.max_concurrent=10 \
    output_dir="logs/full_run"
```

## 相比 MiroFlow 的修改

### 1. 工具系统改造
- **替换搜索工具**: 使用 **Exa AI** 替代原有搜索引擎
- **替换爬虫工具**: 使用 **Firecrawl** 进行网页内容抓取
- **时间约束**: 在工具层面**硬约束检索信源不超过 `end_time`**（FutureX-Past 的历史时间点）
  - 搜索工具添加时间过滤参数
  - 确保模型无法获取未来信息

实现位置：
- `src/tool/mcp_servers/searching_mcp_server_firecrawl_exa.py`
- `src/tool/mcp_servers/reading_mcp_server_firecrawl_exa.py`

### 2. 配置调整
- **Agent 配置**: 参考 MiroFlow 的 `config/agent_gaia-validation-gpt5.yaml`
  - 使用 GPT-5（通过 OpenRouter）
  - 优化参数：`max_turns=6`, `reasoning_effort=low`
- **保持一致**: Pipeline、Orchestrator、LLM Client 等核心流程与 MiroFlow 完全一致

### 3. 评估策略简化
**仅支持 pass@1 评估**（`common_benchmark_past.py`）：
- ❌ 不使用多次投票（majority voting）
- ❌ 不使用大模型验证答案
- ✅ 直接记录 `ground_truth`，输出模型答案和评估结果到 `benchmark_results.jsonl`
- ✅ 方便后续人工验证或脚本验证

输出格式：
```json
{
  "task_id": "task_001",
  "model_boxed_answer": "42",
  "ground_truth": "42",
  "judge_result": "CORRECT",
  "status": "completed"
}
```

## 项目结构

```
futureX_miro/
├── run_futurex_past_fast.py          # 主入口
├── common_benchmark_past.py          # Benchmark 核心（pass@1）
├── config/
│   ├── agent_futurex_past_fast.yaml  # Agent 配置（参考 MiroFlow）
│   ├── benchmark/futurex_past.yaml   # Benchmark 配置
│   └── tool/                          # Exa + Firecrawl 工具配置
└── src/
    ├── core/                          # Pipeline + Orchestrator（保持一致）
    ├── tool/mcp_servers/              # 修改：Exa + Firecrawl 实现
    ├── llm/                           # LLM 客户端（保持一致）
    └── logging/                       # 日志系统（保持一致）
```

## 输出文件

```
logs/full_run/
├── task_{task_id}_attempt_1.json    # 详细执行日志
├── benchmark_results.jsonl          # 评估结果（含 ground_truth）
└── .hydra/config.yaml               # 配置快照
```

## License

Apache-2.0 License

## 致谢

基于 [MiroFlow](https://github.com/MiroMindAI/MiroFlow) 开发，核心流程保持一致。
