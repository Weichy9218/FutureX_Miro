# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0
# 
# Simplified version for Futurex-Past dataset with pass@1 evaluation

import asyncio
import datetime
import json
import os
import signal
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import random

import dotenv
import hydra
import openai
from omegaconf import DictConfig, OmegaConf

from utils.eval_utils import verify_answer_for_datasets
from src.logging.logger import (
    bootstrap_logger,
    task_logging_context,
    init_logging_for_benchmark_evaluation,
)
from config import config_name, config_path
from src.core.pipeline import (
    create_pipeline_components,
    execute_task_pipeline,
)

init_logging_for_benchmark_evaluation(print_task_logs=False)


@dataclass
class BenchmarkTask:
    """Generic benchmark task data structure"""
    task_id: str
    task_question: str
    ground_truth: str
    file_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Generic benchmark evaluation result structure"""
    task_id: str
    task_question: str
    ground_truth: str
    file_path: Optional[str]
    model_response: str
    model_boxed_answer: str
    status: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: str = ""
    judge_result: Optional[str] = None
    log_file_path: Optional[Path] = None

    def to_dict(self):
        """Convert the object to a serializable dictionary."""
        result = self.__dict__.copy()
        # Convert Path objects to string
        if isinstance(result.get("log_file_path"), Path):
            result["log_file_path"] = str(result["log_file_path"])
        if isinstance(result.get("file_path"), Path):
            result["file_path"] = str(result["file_path"])
        return result


class BenchmarkEvaluator:
    """Simplified benchmark evaluator for pass@1"""

    def __init__(self, data_dir: str, benchmark_name: str, cfg: DictConfig):
        """
        Initialize benchmark evaluator

        Args:
            data_dir: Path to benchmark data directory
            benchmark_name: Name of the benchmark
            cfg: The Hydra configuration object
        """
        self.data_dir = Path(data_dir)
        self.benchmark_name = benchmark_name
        self.cfg = cfg
        self.output_dir = Path(cfg.output_dir).absolute()
        if not self.output_dir.exists():
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"Created output directory: {self.output_dir}")
        
        # Initialize evaluation LLM if API key is provided and not "skip_evaluation"
        self.skip_evaluation = (
            cfg.benchmark.openai_api_key == "skip_evaluation" 
            or not cfg.benchmark.openai_api_key
        )
        if not self.skip_evaluation:
            self.evaluation_llm = openai.AsyncOpenAI(api_key=cfg.benchmark.openai_api_key)
        else:
            self.evaluation_llm = None
            print("⚠️  Skipping LLM-based evaluation (openai_api_key is 'skip_evaluation')")
        
        self.tasks: List[BenchmarkTask] = []
        self.results: List[BenchmarkResult] = []

        # Initialize pipeline components
        logs_dir = self.get_log_dir()
        print("Initializing pipeline components...")
        (
            self.main_agent_tool_manager,
            self.sub_agent_tool_managers,
            self.output_formatter,
        ) = create_pipeline_components(cfg, logs_dir=str(logs_dir))
        print("Pipeline components initialized successfully! Using pass@1")

    def load_tasks(self, metadata_file: str) -> List[BenchmarkTask]:
        """
        Load benchmark tasks from metadata.jsonl

        Returns:
            List of BenchmarkTask objects
        """
        metadata_path = self.data_dir / metadata_file
        print(f"Loading tasks from {metadata_path}")

        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        tasks = []
        with open(metadata_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                try:
                    data = json.loads(line.strip())
                    task = BenchmarkTask(
                        task_id=data["task_id"],
                        task_question=data["task_question"],
                        ground_truth=data["ground_truth"],
                        file_path=data.get("file_path"),
                        metadata=data.get("metadata", {}),
                    )
                    
                    # Apply whitelist filter if specified
                    if len(self.cfg.benchmark.data.whitelist) > 0:
                        if task.task_id in self.cfg.benchmark.data.whitelist:
                            tasks.append(task)
                    else:
                        tasks.append(task)

                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line {i + 1}: {e}")
                    continue
        
        # Limit number of tasks if specified
        max_tasks = self.cfg.benchmark.execution.get("max_tasks")
        if max_tasks is not None:
            tasks = tasks[:max_tasks]
        
        self.tasks = tasks
        print(f"Loaded {len(tasks)} tasks")
        return tasks

    def get_log_dir(self) -> Path:
        """Get the log directory for the current benchmark and model."""
        return Path(self.cfg.output_dir)

    def prepare_task_description(
        self, task: BenchmarkTask
    ) -> Tuple[str, Optional[str]]:
        """Prepare task description and file path for the agent"""
        # Build the base task description
        task_description = task.task_question
        
        # Add time constraint information if end_time is available in metadata
        end_time = task.metadata.get("end_time", "")
        if end_time:
            # Extract date for exa_search (format: YYYY-MM-DD)
            end_date = end_time.split('T')[0]
            time_constraint = f"""

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔴 CRITICAL TIME CONSTRAINT - MUST FOLLOW 🔴
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⏰ INFORMATION CUTOFF TIME: {end_time}

You MUST make your prediction based ONLY on information available BEFORE {end_time}.

📋 MANDATORY REQUIREMENTS FOR EACH TOOL:

1️⃣ When using search tools with time filters:
   
   Option A - exa_search (AI-powered search):
   - ALWAYS set: end_published_date="{end_date}"
   - Example: exa_search(q="...", end_published_date="{end_date}")
   
   Option B - firecrawl_search_before (Google-based search with time constraint):
   - ALWAYS set: end_time="{end_date}"
   - Example: firecrawl_search_before(q="...", end_time="{end_date}")
   - This tool AUTOMATICALLY filters results to be before the specified date
   - Recommended for Japanese content or regional searches

2️⃣ 🚫 CRITICAL - NEVER use scrape_website:
   - scrape_website fetches CURRENT webpage content (from {datetime.datetime.today().strftime("%Y-%m-%d")})
   - Current webpages often contain links/content from after {end_time}
   - You MUST use search_archived_webpage instead
   - Extract year, month, day from {end_time}
   - Example: search_archived_webpage(url="...", year={end_time.split('-')[0]}, month={end_time.split('-')[1]}, day={end_time.split('-')[2].split('T')[0]})

3️⃣ For ANY content retrieval:
   - ONLY use: exa_search (with end_published_date), firecrawl_search_before (with end_time), or search_archived_webpage
   - Never use scrape_website under any circumstances

⚠️  VIOLATION WARNING:
Using scrape_website or ANY information published after {end_time} will invalidate your prediction.
This is a strict requirement for fair evaluation.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
            task_description = task_description + time_constraint
        
        # Handle file path
        if task.file_path is None:
            return task_description, None

        path = Path(task.file_path)
        # check if task.file_path is a relative path
        if path.is_absolute():
            return task_description, str(path)

        # Build complete file path: data directory + relative path
        full_file_path = Path(self.data_dir) / path
        return task_description, str(full_file_path)

    def scan_existing_result(self, task: BenchmarkTask) -> Optional[Dict[str, Any]]:
        """Check if task result already exists in filesystem"""
        trace_filename_pattern = f"task_{task.task_id}_attempt_1.json"
        matched_logs = list(self.output_dir.glob(trace_filename_pattern))
        
        if len(matched_logs) == 0:
            return None
        
        latest_log = sorted(matched_logs, reverse=True)[-1]
        print(f"  Found existing log: {latest_log.name}")
        
        try:
            with open(latest_log, "r", encoding="utf-8") as f:
                log_data = json.load(f)
                
                if log_data.get("final_boxed_answer"):
                    return {
                        "model_response": log_data.get("output", ""),
                        "model_boxed_answer": log_data["final_boxed_answer"],
                        "log_file_path": latest_log,
                        "judge_result": log_data.get("judge_result"),
                    }
        except Exception as e:
            print(f"  Error reading existing log: {e}")
        
        return None

    async def run_single_task(self, task: BenchmarkTask) -> BenchmarkResult:
        """
        Run inference for a single benchmark task (pass@1)

        Args:
            task: BenchmarkTask object

        Returns:
            BenchmarkResult object
        """
        print(f"Processing task {task.task_id}")

        result = BenchmarkResult(
            task_id=task.task_id,
            task_question=task.task_question,
            ground_truth=task.ground_truth,
            file_path=task.file_path,
            model_response="",
            model_boxed_answer="",
            status="pending",
            metadata=task.metadata.copy(),
            error_message="",
            judge_result=None,
            log_file_path=None,
        )

        try:
            # Check for existing result
            existing_result = self.scan_existing_result(task)
            
            if existing_result:
                print(f"  ✓ Using cached result: {existing_result['model_boxed_answer']}")
                result.model_response = existing_result["model_response"]
                result.model_boxed_answer = existing_result["model_boxed_answer"]
                result.log_file_path = existing_result["log_file_path"]
                result.status = "completed"
                result.judge_result = existing_result.get("judge_result")
            else:
                # Prepare task
                task_description, task_file_path = self.prepare_task_description(task)

                # Run inference
                print(f"  Running inference...")
                (
                    response,
                    final_boxed_answer,
                    log_file_path,
                ) = await execute_task_pipeline(
                    cfg=self.cfg,
                    task_id=f"{task.task_id}",
                    task_name=f"{task.task_id}",
                    task_file_name=task_file_path,
                    task_description=task_description,
                    main_agent_tool_manager=self.main_agent_tool_manager,
                    sub_agent_tool_managers=self.sub_agent_tool_managers,
                    output_formatter=self.output_formatter,
                    ground_truth=task.ground_truth,
                    metadata=task.metadata,
                    log_path=self.output_dir / f"task_{task.task_id}_attempt_1.json",
                )

                result.model_response = response if response else ""
                result.model_boxed_answer = final_boxed_answer if final_boxed_answer else ""
                result.log_file_path = log_file_path
                result.status = "completed" if final_boxed_answer else "failed"

            # Perform LLM verification if enabled and we have an answer
            if not self.skip_evaluation and result.model_boxed_answer and self.evaluation_llm:
                print(f"  Verifying answer with LLM judge...")
                try:
                    evaluation_result = await verify_answer_for_datasets(
                        openai_client=self.evaluation_llm,
                        benchmark_name=self.benchmark_name,
                        question=task.task_question,
                        target=task.ground_truth,
                        predicted_answer=result.model_boxed_answer,
                        metadata=task.metadata,
                    )
                    result.judge_result = evaluation_result
                    
                    # Update log file with judge result
                    if result.log_file_path:
                        await self._update_log_file_with_evaluation(
                            result.log_file_path, evaluation_result
                        )
                    
                    status_icon = "✅" if evaluation_result == "CORRECT" else "❌"
                    print(f"  {status_icon} Judge result: {evaluation_result}")
                    
                except Exception as e:
                    print(f"  Error verifying answer: {e}")
                    result.judge_result = "ERROR"
            else:
                if self.skip_evaluation:
                    result.judge_result = "SKIPPED"
                    print(f"  ⚠️  Evaluation skipped")

        except Exception as e:
            result.error_message = str(e)
            result.status = "failed"
            print(f"  ❌ Error processing task {task.task_id}: {e}")

        return result

    async def run_parallel_inference(
        self, tasks: List[BenchmarkTask], max_concurrent: int = 3
    ) -> List[BenchmarkResult]:
        """Run inference on multiple tasks in parallel"""
        print(
            f"Running inference on {len(tasks)} tasks with max_concurrent={max_concurrent}"
        )

        semaphore = asyncio.Semaphore(max_concurrent)

        async def run_with_semaphore(task):
            async with semaphore:
                with task_logging_context(task.task_id, self.get_log_dir()):
                    result = await self.run_single_task(task)
                return result

        # Shuffle tasks to avoid order bias
        shuffled_tasks = tasks.copy()
        random.shuffle(shuffled_tasks)

        # Run tasks in parallel
        results = await asyncio.gather(
            *[run_with_semaphore(task) for task in shuffled_tasks],
            return_exceptions=True,
        )

        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Exception in task {shuffled_tasks[i].task_id}: {result}")
                error_result = BenchmarkResult(
                    task_id=shuffled_tasks[i].task_id,
                    task_question=shuffled_tasks[i].task_question,
                    ground_truth=shuffled_tasks[i].ground_truth,
                    file_path=shuffled_tasks[i].file_path,
                    model_response="",
                    model_boxed_answer="",
                    status="failed",
                    metadata=shuffled_tasks[i].metadata.copy(),
                    error_message=str(result),
                    judge_result=None,
                    log_file_path=None,
                )
                processed_results.append(error_result)
            else:
                processed_results.append(result)

        self.results = processed_results
        return processed_results

    def save_results(self, output_path: Path) -> Path:
        """Save evaluation results to JSONL file"""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for result in self.results:
                f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")

        print(f"Results saved to {output_path}")
        return output_path

    def evaluate_accuracy(self) -> float:
        """Evaluate accuracy"""
        if not self.results:
            print("No results to evaluate")
            return 0.0

        print(f"\nCalculating accuracy for {len(self.results)} results...")

        correct_count = 0
        total_count = 0
        evaluated_count = 0

        for result in self.results:
            total_count += 1

            # Display task results
            print(f"\nTask {result.task_id}:")
            print(f"  Status: {result.status}")
            print(f"  Model answer: {result.model_boxed_answer}")
            print(f"  Ground truth: {result.ground_truth}")
            
            if result.judge_result and result.judge_result not in ["SKIPPED", "ERROR"]:
                evaluated_count += 1
                is_correct = result.judge_result == "CORRECT"
                status_icon = "✅" if is_correct else "❌"
                print(f"  Judge result: {status_icon} {result.judge_result}")
                
                if is_correct:
                    correct_count += 1
            else:
                print(f"  Judge result: {result.judge_result or 'NOT_EVALUATED'}")
            
            print("  " + "=" * 50)

        # Calculate accuracy
        if self.skip_evaluation or evaluated_count == 0:
            print(f"\n⚠️  Evaluation was skipped or no tasks were evaluated")
            print(f"Total tasks completed: {total_count}")
            accuracy = 0.0
        else:
            accuracy = correct_count / evaluated_count if evaluated_count > 0 else 0.0
            print(f"\nFinal Results:")
            print(f"Tasks evaluated: {evaluated_count}/{total_count}")
            print(f"Tasks passed: {correct_count}/{evaluated_count}")
            print(f"Accuracy: {accuracy:.2%}")

        return accuracy

    async def _update_log_file_with_evaluation(
        self, log_file_path: Path, evaluation_result: str
    ):
        """Helper method to update log file with evaluation result"""
        try:
            log_file = Path(log_file_path)
            # Read existing data
            with open(log_file, "r", encoding="utf-8") as f:
                log_data = json.load(f)

            # Update with evaluation result
            log_data["judge_result"] = evaluation_result

            # Write to a temporary file and then atomically replace
            temp_log_file = log_file.with_suffix(f"{log_file.suffix}.tmp")
            with open(temp_log_file, "w", encoding="utf-8") as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)

            os.replace(temp_log_file, log_file)
            print(f"    Updated log file with evaluation result.")
        except Exception as e:
            print(f"    Error updating log file {log_file_path}: {e}")


async def entrypoint(cfg: DictConfig) -> float:
    """
    Main entry point for running benchmarks with Hydra.
    """
    print("Benchmark configuration:\n", OmegaConf.to_yaml(cfg, resolve=True))

    evaluator = BenchmarkEvaluator(
        data_dir=cfg.benchmark.data.data_dir,
        benchmark_name=cfg.benchmark.name,
        cfg=cfg,
    )

    """
    Run the full benchmark evaluation process
    """
    print(f"Starting evaluation for benchmark: {cfg.benchmark.name}")

    # Load tasks
    tasks = evaluator.load_tasks(metadata_file=cfg.benchmark.data.metadata_file)
    if len(evaluator.tasks) == 0:
        print("No tasks loaded. Exiting.")
        return 0.0

    # Run inference
    print(
        f"\nStarting parallel inference with {cfg.benchmark.execution.max_concurrent} concurrent tasks..."
    )
    await evaluator.run_parallel_inference(
        tasks,
        max_concurrent=cfg.benchmark.execution.max_concurrent,
    )

    # Evaluate accuracy
    print("\nEvaluating accuracy...")
    accuracy = evaluator.evaluate_accuracy()
    
    # Save results
    output_filename = "benchmark_results.jsonl"
    log_dir = evaluator.output_dir
    results_path = log_dir / output_filename

    evaluator.save_results(results_path)
    print(f"\nEvaluation completed! Results saved to {results_path}")
    
    # save accuracy to a file
    if not evaluator.skip_evaluation:
        accuracy_file = results_path.parent / f"{results_path.stem}_accuracy.txt"
        with open(accuracy_file, "w") as f:
            f.write(f"{accuracy:.2%}")
        print(f"Accuracy saved to {accuracy_file}")

    return accuracy


def setup_hydra_output_dir(cfg: DictConfig, overrides: List[str]) -> DictConfig:
    """Manually creates a Hydra-like output directory and saves the configuration."""
    # Get the base output directory from config
    base_output_dir = Path(cfg.output_dir)

    run_output_dir = base_output_dir
    run_output_dir.mkdir(parents=True, exist_ok=True)

    # Save the composed configuration
    hydra_dir = run_output_dir / ".hydra"
    hydra_dir.mkdir(exist_ok=True)

    with open(hydra_dir / "config.yaml", "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=False))
    with open(hydra_dir / "overrides.yaml", "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(overrides))

    print(f"Hydra-like output directory created at: {run_output_dir}")
    return cfg


def signal_handler(signum, frame):
    """Force exit signal handler"""
    print(f"\n⚠️  Received interrupt signal {signum}, forcing immediate exit...")
    print("Program will terminate all operations immediately")
    os._exit(1)  # Force immediate exit


def main(*args, config_file_name: str = ""):
    # Register signal handlers for immediate response to Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    dotenv.load_dotenv()
    LOGGER_LEVEL = os.getenv("LOGGER_LEVEL", "INFO")

    # Support load from config_file_name
    if config_file_name:
        chosen_config_name = config_file_name
    else:
        chosen_config_name = config_name()

    with hydra.initialize_config_dir(
        config_dir=os.path.abspath(config_path()), version_base=None
    ):
        cfg = hydra.compose(config_name=chosen_config_name, overrides=list(args))
        cfg = setup_hydra_output_dir(cfg, list(args))

        _ = bootstrap_logger(level=LOGGER_LEVEL)
        asyncio.run(entrypoint(cfg))


if __name__ == "__main__":
    import fire
    fire.Fire(main)

