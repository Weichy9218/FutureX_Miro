# 代码结构与依赖关系说明

## 核心模块依赖图

```
run_futurex_past_fast.py
    └─> common_benchmark_past.py
            ├─> src.core.pipeline
            │   ├─> src.llm.client (LLMClient工厂)
            │   │   └─> src.llm.providers.* (具体LLM实现)
            │   │       └─> src.llm.provider_client_base (基类)
            │   ├─> src.tool.manager (ToolManager)
            │   │   └─> src.tool.mcp_servers.* (MCP服务器)
            │   ├─> src.core.orchestrator (Orchestrator)
            │   ├─> src.logging.task_tracer (TaskTracer)
            │   ├─> src.utils.io_utils (OutputFormatter)
            │   └─> src.utils.tool_utils (create_mcp_server_parameters)
            │
            ├─> src.logging.logger (bootstrap_logger)
            ├─> utils.eval_utils (verify_answer_for_datasets)
            └─> config (Hydra配置系统)
```

## 模块详细说明

### 1. 入口层 (Entry Point)

#### `run_futurex_past_fast.py`
- **功能**: 快速版benchmark运行入口
- **依赖**: `common_benchmark_past.main`
- **配置**: 使用 `agent_futurex_past_fast` 配置文件

#### `common_benchmark_past.py`
- **功能**: Benchmark核心逻辑
- **主要类**: 
  - `BenchmarkTask`: 任务数据结构
  - `BenchmarkResult`: 结果数据结构
  - `BenchmarkEvaluator`: 评估器主类
- **关键流程**:
  1. 加载任务 (`load_tasks`)
  2. 并行执行 (`run_parallel_inference`)
  3. 单任务执行 (`run_single_task`)
  4. 评估准确率 (`evaluate_accuracy`)

### 2. 核心层 (Core)

#### `src/core/pipeline.py`
- **功能**: 任务执行管道
- **主要函数**:
  - `execute_task_pipeline()`: 执行完整任务流程
  - `create_pipeline_components()`: 创建管道组件
- **依赖模块**:
  - `LLMClient`: LLM客户端管理
  - `ToolManager`: 工具管理
  - `Orchestrator`: 任务编排
  - `TaskTracer`: 日志追踪

#### `src/core/orchestrator.py`
- **功能**: 协调Agent和工具的交互
- **主要类**: `Orchestrator`
- **关键方法**:
  - `run_main_agent()`: 主Agent执行循环
  - `run_sub_agent()`: 子Agent执行（未使用）
  - `_handle_llm_call_with_logging()`: LLM调用封装
  - `_handle_summary_with_context_limit_retry()`: 摘要生成（含重试）

### 3. 工具层 (Tool)

#### `src/tool/manager.py`
- **功能**: 工具调用管理器
- **主要类**: `ToolManager`
- **关键方法**:
  - `get_all_tool_definitions()`: 获取所有工具定义
  - `execute_tool_call()`: 执行工具调用
  - `_apply_time_constraints()`: 应用时间约束（硬约束）
- **特性**:
  - 支持 StdioServerParameters (MCP协议)
  - 支持 SSE 端点
  - 自动工具查找和纠错
  - Playwright 浏览器会话管理

#### `src/tool/mcp_servers/`
- **searching_mcp_server_firecrawl_exa.py**: 搜索工具服务器
  - `exa_search`: Exa AI搜索
  - `google_search`: Google搜索（通过Firecrawl）
  - `firecrawl_search_before`: 带时间过滤的搜索
  - `search_archived_webpage`: 历史网页搜索
  
- **reading_mcp_server_firecrawl_exa.py**: 阅读工具服务器
  - `scrape_website`: 网页抓取
  - `read_file`: 文件读取（支持多种格式）
  - `read_image`: 图片读取
  
- **browser_session.py**: Playwright浏览器会话管理

### 4. LLM层 (LLM)

#### `src/llm/client.py`
- **功能**: LLM客户端工厂
- **支持的Provider**:
  - GPT5OpenAIClient
  - ClaudeOpenRouterClient

#### `src/llm/provider_client_base.py`
- **功能**: LLM客户端基类
- **抽象方法**:
  - `_create_client()`: 创建具体客户端
  - `_create_message()`: 创建消息
  - `process_llm_response()`: 处理LLM响应
  - `extract_tool_calls_info()`: 提取工具调用信息
  - `update_message_history()`: 更新消息历史

#### `src/llm/providers/gpt5_openai_client.py`
- **功能**: GPT-5 OpenAI客户端
- **特性**:
  - 支持异步调用
  - 支持reasoning_effort参数
  - 自动工具调用解析
  - 上下文限制检测

#### `src/llm/providers/claude_openrouter_client.py`
- **功能**: Claude OpenRouter客户端
- **特性**:
  - Anthropic消息格式
  - Prompt缓存支持
  - 工具使用解析

### 5. 日志层 (Logging)

#### `src/logging/logger.py`
- **功能**: 日志系统配置
- **特性**:
  - ZMQ日志传输（用于工具日志）
  - Rich日志输出
  - 任务上下文追踪
  - Benchmark评估模式

#### `src/logging/task_tracer.py`
- **功能**: 任务执行追踪
- **主要类**: `TaskTracer`
- **记录内容**:
  - 任务信息和状态
  - 主Agent消息历史
  - 子Agent会话历史
  - 步骤日志
  - 最终答案

### 6. 工具函数层 (Utils)

#### `src/utils/io_utils.py`
- **功能**: IO工具
- **主要类/函数**:
  - `process_input()`: 处理输入（文件、任务描述）
  - `OutputFormatter`: 输出格式化
    - `_extract_boxed_content()`: 提取 `\boxed{...}` 内容
    - `format_tool_result_for_user()`: 格式化工具结果
    - `format_final_summary_and_log()`: 格式化最终摘要

#### `src/utils/tool_utils.py`
- **功能**: 工具配置工具
- **主要函数**:
  - `create_mcp_server_parameters()`: 创建MCP服务器参数
  - `expose_sub_agents_as_tools()`: 将子Agent暴露为工具

#### `src/utils/summary_utils.py`
- **功能**: 摘要和答案提取
- **主要函数**:
  - `extract_hints()`: 提取任务提示
  - `extract_gaia_final_answer()`: 提取GAIA格式答案
  - `extract_browsecomp_zh_final_answer()`: 提取中文答案
  - `get_gaia_answer_type()`: 获取答案类型

#### `utils/eval_utils.py`
- **功能**: 答案验证
- **主要函数**:
  - `verify_answer_for_datasets()`: 统一验证接口
  - `verify_answer_gaia()`: GAIA验证器
  - `verify_answer_llm_simpleqa()`: SimpleQA LLM验证
  - `verify_answer_llm_xbench()`: XBench LLM验证
  - `verify_answer_llm_hle()`: HLE LLM验证

### 7. 配置层 (Config)

#### `config/agent_prompts/base_agent_prompt.py`
- **功能**: Agent Prompt基类
- **抽象方法**:
  - `generate_system_prompt_with_mcp_tools()`: 生成系统提示
  - `generate_summarize_prompt()`: 生成摘要提示
  - `expose_agent_as_tool()`: 暴露为工具（子Agent）

#### `config/agent_prompts/main_boxed_answer.py`
- **功能**: 主Agent Prompt实现
- **特性**:
  - 支持中文语境
  - 支持MCP工具
  - Boxed答案格式要求

## 数据流详解

### 1. 任务执行流程

```
1. 加载配置 (Hydra)
2. 创建 BenchmarkEvaluator
3. 加载任务列表 (JSONL)
4. 并行执行任务:
   a. 检查缓存结果
   b. 准备任务描述（添加时间约束）
   c. 执行 execute_task_pipeline:
      - 创建 LLMClient
      - 创建 ToolManager
      - 创建 Orchestrator
      - 设置任务上下文（时间约束）
      - 运行 Orchestrator.run_main_agent():
        * 处理输入（文件、描述）
        * 生成系统提示
        * 主循环：
          · LLM生成响应
          · 解析工具调用
          · 执行工具
          · 更新消息历史
          · 直到无工具调用或达到限制
        * 生成最终摘要
        * 提取最终答案
   d. 验证答案（可选）
5. 保存结果
6. 计算准确率
```

### 2. 工具调用流程

```
1. LLM生成工具调用请求（XML格式）
2. Orchestrator解析工具调用
3. ToolManager.execute_tool_call():
   a. 应用时间约束（如果有）
   b. 连接MCP服务器
   c. 调用工具
   d. 返回结果
4. 格式化结果
5. 更新消息历史
6. 返回给LLM
```

### 3. 时间约束应用（硬约束）

```
1. 任务元数据包含 end_time
2. ToolManager.set_task_context() 设置上下文
3. 工具调用时自动应用：
   - exa_search: 自动添加 end_published_date
   - firecrawl_search_before: 验证 end_time
   - search_archived_webpage: 提取年月日
4. 防止访问未来信息
```

## 关键设计模式

### 1. 工厂模式
- `LLMClient()`: 根据配置创建不同的LLM客户端

### 2. 策略模式
- `LLMProviderClientBase`: 定义统一接口
- 各Provider实现具体策略

### 3. 观察者模式
- `TaskTracer`: 记录任务执行的各个步骤

### 4. 模板方法模式
- `BaseAgentPrompt`: 定义Prompt生成流程
- 子类实现具体内容

## 配置系统 (Hydra)

### 配置层次结构

```
config/
├── agent_futurex_past_fast.yaml    # 主配置
│   ├── defaults:
│   │   └── benchmark: futurex_past
│   ├── main_agent:                  # 主Agent配置
│   │   ├── llm:                     # LLM配置
│   │   └── tool_config:             # 工具配置
│   └── output_dir                   # 输出目录
│
└── benchmark/
    └── futurex_past.yaml            # Benchmark配置
        ├── data:                    # 数据配置
        └── execution:               # 执行配置
```

### 环境变量

所有敏感信息通过环境变量注入：
- `${oc.env:OPENROUTER_API_KEY}`
- `${oc.env:FIRECRAWL_API_KEY}`
- `${oc.env:EXA_API_KEY}`

## 扩展指南

### 添加新的LLM Provider

1. 继承 `LLMProviderClientBase`
2. 实现抽象方法
3. 在 `src/llm/providers/__init__.py` 中注册
4. 在配置中指定 `provider_class`

### 添加新的工具

1. 创建新的MCP服务器文件
2. 在 `config/tool/` 中添加配置
3. 在agent配置中添加到 `tool_config`

### 添加新的Prompt模板

1. 继承 `BaseAgentPrompt`
2. 实现抽象方法
3. 在agent配置中指定 `prompt_class`

## 性能优化

### 已实现的优化

1. **并行执行**: 多任务并发处理
2. **结果缓存**: 跳过已完成任务
3. **工具调用限制**: 防止过度调用
4. **上下文管理**: 定期清理消息历史
5. **异步IO**: LLM和工具调用使用异步
6. **超时控制**: 防止卡死

### 配置参数

- `max_concurrent`: 并发任务数
- `max_turns`: 最大轮次
- `max_tool_calls_per_turn`: 每轮最大工具调用
- `keep_tool_result`: 保留的工具结果数
- `max_tokens`: LLM输出限制
- `reasoning_effort`: 推理深度

## 调试技巧

### 1. 查看任务日志
```bash
cat logs/full_run/task_{task_id}_attempt_1.json | jq
```

### 2. 启用详细日志
```bash
export LOGGER_LEVEL=DEBUG
```

### 3. 单任务测试
```python
python run_futurex_past_fast.py \
    benchmark.execution.max_tasks=1 \
    benchmark.data.whitelist='["task_id_here"]'
```

### 4. 查看工具调用
日志中搜索 "Tool call" 或 "execute_tool_call"

### 5. 检查LLM响应
日志中搜索 "LLM" 或查看 `main_agent_message_history`

