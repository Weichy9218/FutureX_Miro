import json
from pathlib import Path
from datetime import datetime

log_dir = Path('logs/full_run')
task_files = list(log_dir.glob('task_*_attempt_1.json'))

completed = []
running = []
failed = []

for f in task_files:
    try:
        with open(f) as fp:
            data = json.load(fp)
            status = data.get('status', 'unknown')
            if status == 'completed':
                completed.append(f)
            elif status == 'running':
                running.append(f)
            elif status == 'failed':
                failed.append(f)
    except:
        print(f'Error reading {f}')

print(f'当前进度统计:')
print(f'- 总任务数: 851')
print(f'- 已完成: {len(completed)} ({len(completed)/851*100:.1f}%)')
print(f'- 运行中: {len(running)}')
print(f'- 失败: {len(failed)}')

# 计算平均耗时
if completed:
    durations = []
    for f in completed:
        try:
            with open(f) as fp:
                data = json.load(fp)
                start = datetime.fromisoformat(data['start_time'])
                end = datetime.fromisoformat(data['end_time'])
                duration = (end - start).total_seconds()
                durations.append(duration)
        except:
            pass
    
    if durations:
        avg_duration = sum(durations) / len(durations)
        print(f'\n平均耗时: {avg_duration:.1f}秒 ({avg_duration/60:.1f}分钟)')
        
        # 预估剩余时间
        remaining = 851 - len(completed)
        # 默认并发数为5
        max_concurrent = 8
        estimated_time = remaining * avg_duration / max_concurrent
        print(f'预估剩余时间: {estimated_time/3600:.1f}小时')