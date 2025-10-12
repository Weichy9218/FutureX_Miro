#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

import glob
import os
import re
import statistics
import sys


def detect_pass_at_k(results_dir: str) -> tuple:
    """Detect pass_at_k value used in the results directory"""

    # find all possible pass_at_k files
    pattern = os.path.join(
        results_dir, "run_*", "benchmark_results_pass_at_*_accuracy.txt"
    )
    all_files = glob.glob(pattern)

    if not all_files:
        print(f"No accuracy files found in {results_dir}")
        print(f"Expected pattern: {pattern}")
        return None, []

    # get value `k` from the first such file
    filename = os.path.basename(all_files[0])
    match = re.search(r"pass_at_(\d+)_accuracy\.txt", filename)

    if not match:
        print(f"Cannot extract pass_at_k from filename: {filename}")
        return None, []

    k = int(match.group(1))

    # get all file with the same `k` value
    accuracy_files = glob.glob(
        os.path.join(
            results_dir, "run_*", f"benchmark_results_pass_at_{k}_accuracy.txt"
        )
    )

    return k, accuracy_files


def calculate_average_scores(results_dir: str) -> dict:
    """Calculate average scores across multiple runs - automatically detect pass_at_k value"""

    # return all accuracy_files with same `pass_at_k` value
    pass_at_k, accuracy_files = detect_pass_at_k(results_dir)

    if pass_at_k is None:
        return {}

    print(f"Detected pass_at_{pass_at_k} files")
    print(f"Found {len(accuracy_files)} accuracy files")

    scores = []

    # Read each accuracy file
    for i, file_path in enumerate(sorted(accuracy_files), 1):
        try:
            with open(file_path, "r") as f:
                content = f.read().strip()
                # Remove percentage sign and convert to float
                score = float(content.replace("%", ""))
                scores.append(score)
                print(f"Run {i}: {score:.2f}%")
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

    if not scores:
        print("No valid scores found")
        return {}

    # Calculate statistics
    stats = {
        "pass_at_k": pass_at_k,
        "num_runs": len(scores),
        "individual_scores": scores,
        "average_score": statistics.mean(scores),
        "std_dev": statistics.stdev(scores) if len(scores) > 1 else 0,
        "min_score": min(scores),
        "max_score": max(scores),
    }

    return stats


def print_results(stats: dict):
    """Print results"""
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)

    print(f"Pass@{stats['pass_at_k']} Results:")
    print(f"Number of runs: {stats['num_runs']}")
    print(f"Individual scores: {[f'{s:.2f}%' for s in stats['individual_scores']]}")
    print()
    print(f"Standard deviation: {stats['std_dev']:.2f}%")
    print(f"Min score: {stats['min_score']:.2f}%")
    print(f"Max score: {stats['max_score']:.2f}%")
    print(f"Average score: {stats['average_score']:.2f}%")
    print("=" * 50)


def main(results_dir: str):
    stats = calculate_average_scores(results_dir)

    if stats:
        print_results(stats)

        # save statistics to file
        output_file = os.path.join(
            results_dir, f"average_scores_pass_at_{stats['pass_at_k']}.txt"
        )
        with open(output_file, "w") as f:
            f.write("EVALUATION RESULTS\n")
            f.write("=" * 50 + "\n")
            f.write(f"Pass@{stats['pass_at_k']} Results:\n")
            f.write(f"Number of runs: {stats['num_runs']}\n")
            f.write(
                f"Individual scores: {[f'{s:.2f}%' for s in stats['individual_scores']]}\n"
            )
            f.write(f"Standard deviation: {stats['std_dev']:.2f}%\n")
            f.write(f"Min score: {stats['min_score']:.2f}%\n")
            f.write(f"Max score: {stats['max_score']:.2f}%\n")
            f.write(f"Average score: {stats['average_score']:.2f}%\n")
            f.write("=" * 50 + "\n")

        print(f"\nResults saved to: {output_file}")
    else:
        print("Failed to calculate statistics")
        sys.exit(1)
