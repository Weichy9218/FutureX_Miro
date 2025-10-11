# 快速启动指南

## 1. 环境准备

### 安装依赖
```bash
pip install -r requirements.txt
```

### 配置环境变量
```bash
# 复制示例文件
cp .env.example .env

# 编辑 .env 文件，填入你的API密钥
nano .env
```

必需的API密钥：
- `OPENROUTER_API_KEY`: OpenRouter API密钥（用于GPT-5）
- `FIRECRAWL_API_KEY`: Firecrawl API密钥（网页抓取）
- `EXA_API_KEY`: Exa API密钥（AI搜索）

## 2. 准备数据集

```bash
# 创建数据目录
mkdir -p data/futurex

# 将 FutureX-Past 数据集放入该目录
# 确保有以下文件：
# data/futurex/standardized_data_past.jsonl
```

## 3. 运行测试

### 测试单个任务
```bash
python run_futurex_past_fast.py \
    benchmark.execution.max_tasks=1 \
    output_dir="logs/test"
```

### 查看结果
```bash
# 查看任务日志
ls logs/test/

# 查看详细日志（需要安装jq）
cat logs/test/task_*_attempt_1.json | jq
```

## 4. 运行完整评估

### 小规模测试（10个任务）
```bash
python run_futurex_past_fast.py \
    benchmark.execution.max_tasks=10 \
    benchmark.execution.max_concurrent=3 \
    output_dir="logs/test_10"
```

### 中等规模（100个任务）
```bash
python run_futurex_past_fast.py \
    benchmark.execution.max_tasks=100 \
    benchmark.execution.max_concurrent=5 \
    output_dir="logs/test_100"
```

### 完整评估（所有任务）
```bash
python run_futurex_past_fast.py \
    benchmark.execution.max_tasks=851 \
    benchmark.execution.max_concurrent=10 \
    output_dir="logs/full_run"
```

## 5. 查看结果

### 评估摘要
```bash
# 查看准确率（如果启用了评估）
cat logs/full_run/benchmark_results_accuracy.txt

# 查看所有结果
cat logs/full_run/benchmark_results.jsonl
```

### 单任务分析
```bash
# 使用jq查看结构化日志
cat logs/full_run/task_{task_id}_attempt_1.json | jq '{
  task_id: .task_id,
  status: .status,
  final_answer: .final_boxed_answer,
  judge_result: .judge_result
}'
```

## 6. 常见配置调整

### 调整并发数
```bash
# 减少并发（节省资源）
benchmark.execution.max_concurrent=3

# 增加并发（加快速度）
benchmark.execution.max_concurrent=15
```

### 调整轮次限制
```bash
# 减少轮次（加快速度）
main_agent.max_turns=4

# 增加轮次（提高准确率）
main_agent.max_turns=10
```

### 调整工具调用限制
```bash
# 每轮工具调用次数
main_agent.max_tool_calls_per_turn=8
```

### 启用LLM评估
编辑配置文件或命令行：
```bash
benchmark.openai_api_key="your_openai_api_key"
```

## 7. 故障排查

### API密钥错误
```bash
# 检查环境变量
echo $OPENROUTER_API_KEY

# 重新加载环境变量
source .env
export $(cat .env | xargs)
```

### 数据集路径错误
```bash
# 检查数据文件
ls -lh data/futurex/standardized_data_past.jsonl

# 调整路径
export DATA_DIR=/path/to/your/data
```

### 内存不足
```bash
# 减少并发数
benchmark.execution.max_concurrent=2

# 限制任务数
benchmark.execution.max_tasks=50
```

### 查看详细日志
```bash
# 启用DEBUG日志
export LOGGER_LEVEL=DEBUG

# 运行
python run_futurex_past_fast.py ...
```

## 8. 高级用法

### 指定特定任务
```bash
python run_futurex_past_fast.py \
    'benchmark.data.whitelist=["task_id_1","task_id_2"]'
```

### 修改输出token限制
```bash
python run_futurex_past_fast.py \
    main_agent.llm.max_tokens=32000
```

### 修改推理深度
```bash
# 快速模式
main_agent.llm.reasoning_effort=low

# 中等模式
main_agent.llm.reasoning_effort=medium

# 深度模式
main_agent.llm.reasoning_effort=high
```

## 9. 性能优化建议

### 快速评估（牺牲准确率）
```bash
python run_futurex_past_fast.py \
    main_agent.max_turns=4 \
    main_agent.max_tool_calls_per_turn=4 \
    main_agent.llm.max_tokens=8000 \
    main_agent.llm.reasoning_effort=low \
    benchmark.execution.max_concurrent=15
```

### 高准确率（牺牲速度）
```bash
python run_futurex_past_fast.py \
    main_agent.max_turns=10 \
    main_agent.max_tool_calls_per_turn=10 \
    main_agent.llm.max_tokens=32000 \
    main_agent.llm.reasoning_effort=high \
    benchmark.execution.max_concurrent=3
```

## 10. 监控运行

### 实时监控日志
```bash
# 监控输出目录
watch -n 5 'ls -lh logs/full_run/*.json | wc -l'

# 监控最新任务
tail -f logs/full_run/task_logs/task_*.log
```

### 检查完成进度
```bash
# 统计完成的任务
grep -l "completed" logs/full_run/task_*.json | wc -l

# 统计失败的任务
grep -l "failed" logs/full_run/task_*.json | wc -l
```

## 下一步

- 查看 `README.md` 了解项目详情
- 查看 `CODE_STRUCTURE.md` 了解代码架构
- 根据需要调整 `config/agent_futurex_past_fast.yaml` 配置
