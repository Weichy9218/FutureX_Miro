#!/usr/bin/env python3
"""
提取Futurex-Past结果并生成CSV文件
提取字段: task_id, task_description, end_time, level, final_boxed_answer, ground_truth
"""

import json
import csv
import re
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

def extract_boxed_answer(text: str) -> str:
    """从文本中提取\\boxed{...}中的内容，从后往前匹配最后一个boxed"""
    if not text:
        return ""
    
    # 匹配 \boxed{...} 格式，使用findall找到所有匹配
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = re.findall(pattern, text)
    
    # 如果有匹配，返回最后一个（即从后往前第一个）
    if matches:
        return matches[-1].strip()
    return ""

def process_json_files(directory: str) -> List[Dict]:
    """处理目录中的所有JSON文件，提取关键信息"""
    results = []
    dir_path = Path(directory)
    
    # 获取所有JSON文件
    json_files = list(dir_path.glob("task_*_attempt_1.json"))
    print(f"找到 {len(json_files)} 个任务文件")
    
    for file_path in json_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
                # 提取基本信息
                task_id = data.get("task_id", "")
                task_description = data.get("input", {}).get("task_description", "")
                
                # 筛选task_description，只保留CRITICAL TIME CONSTRAINT之前的部分
                critical_marker = "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n🔴 CRITICAL TIME CONSTRAINT - MUST FOLLOW"
                if critical_marker in task_description:
                    task_description = task_description.split(critical_marker)[0].strip()

                # 提取metadata
                metadata = data.get("input", {}).get("metadata", {})
                end_time = metadata.get("end_time", "")
                level = metadata.get("level", "")
                
                # 提取答案
                final_answer = ""
                # 从step_logs中查找final_answer_content
                step_logs = data.get("step_logs", [])
                for step in step_logs:
                    if step.get("step_name") == "final_answer_content":
                        final_answer = step.get("message", "")
                        break
                
                boxed_content = extract_boxed_answer(final_answer)
                if not boxed_content:
                    comleted_status = "failed"
                else:
                    comleted_status = "completed"
                
                # 提取ground truth
                ground_truth = data.get("ground_truth", "")
                
                # 计算耗时
                start_time = data.get("start_time", "")
                end_time_exec = data.get("end_time", "")
                duration = ""
                if start_time and end_time_exec:
                    try:
                        start = datetime.fromisoformat(start_time)
                        end = datetime.fromisoformat(end_time_exec)
                        duration = str((end - start).total_seconds())
                    except:
                        pass

                # 添加到结果列表
                results.append({
                    "task_id": task_id,
                    "task_description": task_description,
                    "end_time": end_time,
                    "level": level,
                    "final_answer": final_answer,
                    "boxed_content": boxed_content,
                    "ground_truth": ground_truth,
                    "duration_seconds": duration,
                    "completion_status": comleted_status
                })
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
    
    return results

def save_to_csv(results: List[Dict], output_file_path: str):
    """将结果保存为CSV文件，仅保存已完成的任务"""
    if not results:
        print("没有结果可保存")
        return
    
    # 筛选出已完成的任务
    completed_results = [r for r in results if r.get("completion_status") == "completed"]
    
    if not completed_results:
        print("没有已完成的任务结果可保存")
        return
    
    # 定义CSV列顺序
    fieldnames = [
        "task_id", "task_description", "level", "end_time",
        "ground_truth", "final_answer", "boxed_content", "duration_seconds", 
        "completion_status"
    ]
    
    with open(output_file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(completed_results)
    
    print(f"结果已保存到: {output_file_path}")
    print(f"共保存了 {len(completed_results)} 条已完成的任务结果")

def main():
    """主函数"""
    # 设置输入和输出路径
    input_dir = "logs/futurex_past_fast"
    output_file = "futurex_past_fast_results.csv"
    output_file_path = os.path.join(input_dir, output_file)
    
    # 处理文件
    print(f"正在处理目录: {input_dir}")
    results = process_json_files(input_dir)
    
    # 保存结果
    save_to_csv(results, output_file_path)
    
    # 打印统计信息
    # 根据是否有boxed_content来判断是否完成
    completed = sum(1 for r in results if r["completion_status"] == "completed")
    not_completed = len(results) - completed
    
    print("\n统计信息:")
    print(f"- 总任务数: {len(results)}")
    print(f"- 已完成(有boxed结果): {completed}")
    print(f"- 未完成/进行中: {not_completed}")
    
    if completed > 0:
        # 计算平均耗时
        durations = [float(r["duration_seconds"]) for r in results if r["boxed_content"] and r["duration_seconds"]]
        if durations:
            avg_duration = sum(durations) / len(durations)
            print(f"\n平均耗时: {avg_duration:.1f}秒 ({avg_duration/60:.1f}分钟)")

if __name__ == "__main__":
    main()
