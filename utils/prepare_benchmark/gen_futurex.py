# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Generator, MutableMapping

from datasets import load_dataset

from utils.prepare_benchmark.common import Task


def gen_futurex(hf_token: str) -> Generator[Task, None, None]:
    """
    Generate Futurex-Online dataset tasks in MiroFlow format

    Args:
        hf_token: Hugging Face token for dataset access

    Yields:
        Task: Standardized task objects
    """
    # Load the Futurex-Online dataset
    # dataset = load_dataset("futurex-ai/Futurex-Online")
    dataset = load_dataset("futurex-ai/Futurex-Past")

    # Process each split in the dataset
    for split_name, split_data in dataset.items():
        for idx, sample in enumerate(split_data):
            # Extract task information
            task_id = sample.get("question_id", f"futurex_{split_name}_{idx}")  # id
            task_question = sample.get("prompt", "")
            end_time = sample.get("end-time", "")
            level = sample.get("level", "")
            ground_truth = sample.get("answer", "")

            # Create metadata dictionary
            metadata: MutableMapping = {
                "level": level,
                "end_time": end_time,
                "source": "futurex-ai/Futurex-Past",  # Futurex-Online
                "split": split_name,
                "original_id": sample.get("id", ""),
                "dataset_name": "Futurex-Past", # Online
            }

            # Create standardized Task object
            task = Task(
                task_id=task_id,
                task_question=task_question,
                ground_truth=ground_truth,  # Futurex-Online doesn't have ground truth
                file_path=None,  # No file attachments
                metadata=metadata,
            )

            yield task

    return
