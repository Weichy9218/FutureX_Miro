#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

"""
下载和准备 FutureX-Past 数据集
"""

import os
import pathlib
import dotenv
from utils.prepare_benchmark.gen_futurex import gen_futurex

def main():
    """下载 FutureX-Past 数据集"""
    # 加载环境变量
    dotenv.load_dotenv()
    
    # 获取配置
    data_dir = os.getenv("DATA_DIR", "data")
    hf_token = os.getenv("HF_TOKEN", "")
    
    if not hf_token:
        print("错误: 请在 .env 文件中设置 HF_TOKEN")
        print("获取 token: https://huggingface.co/settings/tokens")
        return
    
    # 创建数据目录
    data_path = pathlib.Path(data_dir) / "futurex"
    data_path.mkdir(parents=True, exist_ok=True)
    
    meta_file = data_path / "standardized_data_past.jsonl"
    
    print(f"开始下载 FutureX-Past 数据集...")
    print(f"数据将保存到: {meta_file}")
    
    # 生成数据集
    task_count = 0
    with open(meta_file, "wb") as f:
        for task in gen_futurex(hf_token):
            f.write(task.to_json())
            f.write(b"\n")
            task_count += 1
            if task_count % 100 == 0:
                print(f"已处理 {task_count} 个任务...")
    
    print(f"\n✓ 完成! 共处理 {task_count} 个任务")
    print(f"数据集已保存到: {meta_file}")
    print(f"\n现在可以运行: python run_futurex_past_fast.py")

if __name__ == "__main__":
    main()

