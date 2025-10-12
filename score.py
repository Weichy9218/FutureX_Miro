#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluation script for FutureAgent predictions

Scoring rules (from the provided paper snippet):
- Level 1 (single-choice): 0/1 accuracy.
- Level 2 (multi-choice): F1 score between predicted set and ground-truth set.
- Level 3 (open-ended ranking top-k):
  score = 1 if order matches exactly; else 0.8 * |overlap| / k.
- Level 4 (open-ended numerical):
  score = max(0, 1 - ((Y - Y_hat)/sigma)^2). If sigma is missing or <= 0,
  gracefully fall back to exact match (1 if equal, else 0).

Only the content inside "\\boxed{...}" from the model output column is used.

Usage example:
  python evaluate.py \
    --csv /home/chuyangwei/FutureAgent/predictions_grok4.csv \
    --prediction-column grok4_prediction

Optional for level-4 numerical tasks:
  - Provide a sigma column in the same CSV via --sigma-column
  - Or provide a separate mapping file with columns [question_id, sigma]
    via --sigma-csv
  - Or use --sigma-default (default 1.0 here) to apply a constant sigma

This script prints overall average score and per-level averages, and writes
per-row scores to an output CSV if --out is provided.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Iterable, List, Optional, Sequence, Tuple, Dict
import pandas as pd

# Add parent directory to path to import from utils/evaluate.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.evaluate import (
    extract_boxed_text,
    literal_eval_list,
    split_items_for_level,
    normalize_token,
    score_level1,
    score_level2,
    score_level3,
    score_level4,
    try_parse_float
)


def build_sigma_map(sigma_csv: Optional[str]) -> dict:
    """Load sigma values from CSV with columns: question_id, sigma."""
    if not sigma_csv or not os.path.exists(sigma_csv):
        return {}
    df = pd.read_csv(sigma_csv)
    if "question_id" not in df.columns or "sigma" not in df.columns:
        raise ValueError("sigma-csv must contain columns: question_id, sigma")
    return {str(r["question_id"]): float(r["sigma"]) 
            for _, r in df.iterrows() if pd.notna(r["sigma"])}


def load_alias_map(alias_csv: Optional[str]) -> Dict[str, str]:
    """Load alias map from CSV with columns: alias, canonical."""
    if not alias_csv or not os.path.exists(alias_csv):
        return {}
    df = pd.read_csv(alias_csv)
    if not {"alias", "canonical"}.issubset(set(df.columns)):
        raise ValueError("alias-csv must contain columns: alias, canonical")
    result = {}
    for _, r in df.iterrows():
        alias_norm = normalize_token(r.get("alias"))
        if alias_norm:
            result[alias_norm] = normalize_token(r.get("canonical"))
    return result


def detect_lang(text: str) -> str:
    """Detect language: CJK -> 'zh', otherwise 'en'."""
    return "zh" if text and re.search(r"[\u3400-\u9FFF]", text) else "en"


_translator = None


def try_translate(text: str, target_lang: str) -> str:
    """Best-effort translation; fallback to original text if fails."""
    global _translator
    if not text:
        return text
    try:
        if _translator is None:
            from googletrans import Translator  # type: ignore
            _translator = Translator()
        res = _translator.translate(text, dest=target_lang)
        return str(res.text).strip() if getattr(res, "text", None) else text
    except Exception:
        return text


def canonicalize_with_alias(items: List[str], alias_map: Dict[str, str]) -> List[str]:
    """Apply alias mapping to normalize items."""
    if not alias_map:
        return items
    return [alias_map.get(normalize_token(x), normalize_token(x)) for x in items]


def evaluate(
    csv_path: str,
    prediction_column: str = "grok4_prediction",
    ground_truth_column: str = "answer",
    level_column: str = "level",
    sigma_column: Optional[str] = None,
    sigma_csv: Optional[str] = None,
    sigma_default: float = 1.0,
    translate_level3: bool = False,
    alias_csv: Optional[str] = None,
    use_relative_sigma: bool = True,
    relative_sigma_ratio: float = 0.1,
) -> Tuple[pd.DataFrame, dict]:
    df = pd.read_csv(csv_path)
    if prediction_column not in df.columns:
        raise ValueError(f"Prediction column '{prediction_column}' not found in CSV")
    if ground_truth_column not in df.columns:
        raise ValueError(f"Ground truth column '{ground_truth_column}' not found in CSV")
    if level_column not in df.columns:
        raise ValueError(f"Level column '{level_column}' not found in CSV")

    sigma_map = build_sigma_map(sigma_csv)
    alias_map = load_alias_map(alias_csv)

    parsed_ok: List[bool] = []
    out_rows = []
    
    for _, row in df.iterrows():
        qid = str(row.get("task_id", ""))
        level = int(row[level_column]) if pd.notna(row[level_column]) else None
        gt_list = literal_eval_list(row[ground_truth_column])
        
        # Extract and parse prediction
        boxed = extract_boxed_text(row[prediction_column])
        pred_items = split_items_for_level(boxed, level) if boxed else []
        parsed_ok.append(bool(boxed and pred_items))

        # Get sigma value for numerical tasks
        sigma_val = None
        if sigma_column and pd.notna(row.get(sigma_column)):
            try:
                sigma_val = float(row[sigma_column])
            except Exception:
                pass
        if sigma_val is None:
            sigma_val = sigma_map.get(qid)
        # Use sigma_default only when not using relative sigma
        if sigma_val is None and not use_relative_sigma:
            sigma_val = sigma_default

        # Calculate score based on level
        if level == 1:
            score = score_level1(gt_list, pred_items)
        elif level == 2:
            score = score_level2(gt_list, pred_items)
        elif level == 3:
            # Use numerical scoring if ground truth is a single number
            is_numerical = (gt_list and len(gt_list) == 1 and 
                          try_parse_float(gt_list[0]) is not None)
            if is_numerical:
                score = score_level4(gt_list, pred_items, sigma_val, 
                                   use_relative_sigma, relative_sigma_ratio)
            else:
                # Ranking task with optional translation
                if translate_level3:
                    tgt_lang = detect_lang(" ".join(str(x) for x in gt_list))
                    pred_items = [try_translate(x, tgt_lang) for x in pred_items]
                    pred_items = canonicalize_with_alias(pred_items, alias_map)
                    gt_list = canonicalize_with_alias([normalize_token(x) for x in gt_list], alias_map)
                score = score_level3(gt_list, pred_items)
        elif level == 4:
            score = score_level4(gt_list, pred_items, sigma_val, 
                               use_relative_sigma, relative_sigma_ratio)
        else:
            score = 0.0

        out_rows.append({
            "task_id": qid,
            "level": level,
            "score": score,
            "boxed_pred": boxed,
        })

    # Build summary statistics
    result_df = pd.DataFrame(out_rows)
    overall = float(result_df["score"].mean()) if not result_df.empty else 0.0
    
    # Calculate per-level statistics
    level_stats = {}
    total_count = len(result_df)
    if not result_df.empty:
        level_groups = result_df.groupby("level")
        for level, group in level_groups:
            level_count = len(group)
            level_stats[int(level)] = {
                "score": float(group["score"].mean()),
                "count": level_count,
                "percentage": level_count / total_count if total_count > 0 else 0.0
            }
    
    summary = {
        "overall": overall,
        "level_stats": level_stats,
        "num_rows": total_count,
        "parsed_rate": sum(parsed_ok) / len(parsed_ok) if parsed_ok else 0.0,
    }
    return result_df, summary


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate predictions with level-specific metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--csv", required=True, help="Path to predictions CSV")
    parser.add_argument("--prediction-column", default="grok4_prediction", help="Prediction column name")
    parser.add_argument("--ground-truth-column", default="answer", help="Ground-truth column name")
    parser.add_argument("--level-column", default="level", help="Level column name")
    parser.add_argument("--sigma-column", default=None, help="Sigma column for numerical tasks")
    parser.add_argument("--sigma-csv", default=None, help="CSV with question_id,sigma mapping")
    parser.add_argument("--sigma-default", type=float, default=1.0, help="Default sigma (used when --no-relative-sigma)")
    parser.add_argument("--use-relative-sigma", action="store_true", default=True, 
                       help="Use relative sigma = ratio * |ground_truth| (default: True)")
    parser.add_argument("--no-relative-sigma", dest="use_relative_sigma", action="store_false", 
                       help="Use fixed sigma instead of relative")
    parser.add_argument("--relative-sigma-ratio", type=float, default=0.1, 
                       help="Relative sigma ratio (default: 0.1 = 10%%)")
    parser.add_argument("--translate-level3", action="store_true", 
                       help="Translate level-3 predictions to ground truth language")
    parser.add_argument("--alias-csv", default=None, help="CSV with alias,canonical mapping")
    parser.add_argument("--out", default=None, help="Output CSV path for per-row scores")
    args = parser.parse_args()

    result_df, summary = evaluate(
        csv_path=args.csv,
        prediction_column=args.prediction_column,
        ground_truth_column=args.ground_truth_column,
        level_column=args.level_column,
        sigma_column=args.sigma_column,
        sigma_csv=args.sigma_csv,
        sigma_default=args.sigma_default,
        translate_level3=args.translate_level3,
        alias_csv=args.alias_csv,
        use_relative_sigma=args.use_relative_sigma,
        relative_sigma_ratio=args.relative_sigma_ratio,
    )
    
    if args.out:
        result_df.to_csv(args.out, index=False)

    # Print summary
    print("=== Evaluation Summary ===")
    print(f"Total rows     : {summary['num_rows']}")
    print(f"Parsed rate    : {summary['parsed_rate']:.3f}")
    print(f"Overall score  : {summary['overall']:.6f}")
    print("\nPer-level statistics:")
    print(f"{'Level':<8} {'Count':<8} {'Percent':<10} {'Score':<10}")
    print("-" * 40)
    for lvl in sorted(summary["level_stats"].keys()):
        stats = summary["level_stats"][lvl]
        print(f"{lvl:<8} {stats['count']:<8} {stats['percentage']*100:>6.2f}%    {stats['score']:.6f}")


if __name__ == "__main__":
    main()