#!/usr/bin/env python3
"""
快速版 Futurex-Past benchmark 运行脚本
优化了搜索策略和资源限制，以加快处理速度
"""

# 导入标准的 benchmark 运行逻辑
from common_benchmark_past import main

if __name__ == "__main__":
    import fire
    # 使用快速配置文件
    fire.Fire(lambda *args: main(*args, config_file_name="agent_futurex_past_fast"))

