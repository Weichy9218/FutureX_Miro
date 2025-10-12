#!/usr/bin/env python3
"""
查找状态不为completed的任务并输出task_id
"""

import json
import os
from pathlib import Path

def find_incomplete_tasks(directory: str):
    """
    查找状态不为completed的任务
    
    Args:
        directory: 日志目录
    """
    dir_path = Path(directory)
    
    # 获取所有JSON文件
    json_files = list(dir_path.glob("task_*_attempt_1.json"))
    
    incomplete_task_ids = []
    
    for file_path in json_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
                task_id = data.get("task_id", "")
                status = data.get("status", "")
                
                if status != "completed":
                    incomplete_task_ids.append(task_id)
                    
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
    
    return incomplete_task_ids

def main():
    # 设置输入目录
    input_dir = "/home/chuyangwei/MiroFlow/logs/futurex_past_fast"
    
    print(f"正在检查目录: {input_dir}")
    print(len(os.listdir(input_dir)))
    
    # 查找未完成的任务
    incomplete_tasks = find_incomplete_tasks(input_dir)
    
    # 输出结果
    if incomplete_tasks:
        print(f"\n找到 {len(incomplete_tasks)} 个未完成的任务:")
        for task_id in incomplete_tasks:
            print(task_id)
    else:
        print("\n所有任务都已完成")

if __name__ == "__main__":
    main()
