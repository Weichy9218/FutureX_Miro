#!/usr/bin/env python3
import json
import os
import glob
import shutil
import argparse

def find_and_handle_error_tasks(delete_files=False, error_type="[ERROR]"):
    """
    查找并处理错误任务
    
    参数:
        delete_files: 是否删除错误任务文件
        error_type: 错误类型，可选 "[ERROR]" 或 "failed to complete"
    """
    log_dir = "/home/chuyangwei/MiroFlow/logs/futurex_past_fast"
    error_tasks = []
    deleted_files = []
    
    # 获取所有JSON文件
    json_files = glob.glob(os.path.join(log_dir, "*.json"))
    print(f"找到 {len(json_files)} 个JSON文件")
    
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 获取task_id
            task_id = None
            if isinstance(data, dict):
                task_id = data.get("task_id")
                
                # 检查是否包含错误
                has_error = False
                
                # 检查 final_answer_content 中的错误
                step_logs = data.get("step_logs", [])
                if isinstance(step_logs, list):
                    for step in step_logs:
                        message = step.get("message", "")
                        if step.get("step_name", "") == "final_answer_content" and error_type in str(message):
                            has_error = True
                            error_message = message
                
                # 如果找到错误
                if has_error:
                    error_tasks.append({
                        "task_id": task_id,
                        "file_path": file_path,
                        "error_message": error_message[:200] + "..." if len(error_message) > 200 else error_message
                    })
                    
                    # 如果需要删除文件
                    if delete_files:
                        try:
                            os.remove(file_path)
                            deleted_files.append(file_path)
                        except Exception as e:
                            print(f"删除文件 {file_path} 失败: {e}")

        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
    
    return error_tasks, deleted_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="查找并处理错误任务")
    parser.add_argument("--delete", action="store_true", help="删除错误任务文件")
    parser.add_argument("--error-type", choices=["[ERROR]", "failed"], default="[ERROR]", 
                       help="错误类型: [ERROR] 或 failed (failed to complete)")
    args = parser.parse_args()
    
    # 转换错误类型
    error_type = "[ERROR]" if args.error_type == "[ERROR]" else "failed to complete"
    
    # 查找并处理错误任务
    error_tasks, deleted_files = find_and_handle_error_tasks(args.delete, error_type)
    
    # 打印结果
    print("\n找到的错误任务:")
    print("=" * 50)
    for i, task in enumerate(error_tasks, 1):
        print(f"{i}. Task ID: {task['task_id']}")
        print(f"   文件: {os.path.basename(task['file_path'])}")
        print(f"   错误: {task['error_message']}")
        print("-" * 50)
    
    print(f"\n总计包含 {args.error_type} 的task数量: {len(error_tasks)}")
    
    if args.delete:
        print(f"\n已删除 {len(deleted_files)} 个错误任务文件")
        print("这些任务现在可以重新运行了")
