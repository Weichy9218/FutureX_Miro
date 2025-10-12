import re
import ast
import math
from typing import Iterable, List, Optional, Sequence, Tuple, Dict

BOXED_PATTERN = re.compile(r"\\boxed\{([^}]*)\}", flags=re.DOTALL)

def extract_boxed_text(text: str) -> Optional[str]:
    """Extract the last occurrence of content inside \\boxed{...}."""
    if not text:
        return None
    try:
        matches = BOXED_PATTERN.findall(str(text))
        return matches[-1].strip() if matches else None
    except Exception:
        return None


def literal_eval_list(s: str) -> List:
    """Parse answer column stored as "['A']" or similar format."""
    if not s:
        return []
    if isinstance(s, (list, tuple)):
        return list(s)
    text = str(s).strip()
    try:
        val = ast.literal_eval(text)
        return list(val) if isinstance(val, (list, tuple)) else [val]
    except Exception:
        # Fallback: split by common delimiters
        parts = re.split(r"[\s,，、]+", text)
        return [p for p in (x.strip() for x in parts) if p]



def split_items_for_level(content: str, level: int) -> List[str]:
    """Split boxed content based on level type.
    
    - Level 4: Return raw string (numerical)
    - Level 1/2/3: Split by delimiters and normalize
    """
    if not content:
        return []
    s = str(content).strip()
    if level == 4:
        return [s]
    
    # Split by common delimiters
    parts = re.split(r"[\n,;，、]+", s)
    # Handle numbered items in same line: "1. A  2. B  3. C"
    if len(parts) == 1:
        numbered = re.split(r"\s(?=\d+[\.)、．。])", s)
        if len(numbered) > 1:
            parts = numbered
    
    return [normalize_token(p) for p in parts if p and normalize_token(p)]



def normalize_token(x: str) -> str:
    """Normalize token: remove numbering, collapse spaces, uppercase single letters/yes/no."""
    if not x:
        return ""
    s = str(x).strip()
    s = re.sub(r"^\s*\d+[\.)、．。]\s*", "", s)  # Remove leading numbering
    s = re.sub(r"\s+", " ", s).strip()  # Collapse spaces
    # Uppercase single letters and yes/no
    if re.fullmatch(r"[A-Z]", s, flags=re.IGNORECASE) or s.lower() in {"yes", "no"}:
        return s.upper()
    return s


def to_set(items: Sequence[str]) -> List[str]:
    """Return unique items preserving first occurrence order."""
    seen = set()
    result = []
    for x in items:
        if x not in seen:
            seen.add(x)
            result.append(x)
    return result


def f1_score_sets(gt: Sequence[str], pred: Sequence[str]) -> float:
    """Calculate F1 score between two sets of strings."""
    gt_set = set(to_set(gt))
    pred_set = set(to_set(pred))
    
    if not gt_set and not pred_set:
        return 1.0
    if not gt_set or not pred_set:
        return 0.0
    
    tp = len(gt_set & pred_set)
    if tp == 0:
        return 0.0
    
    precision = tp / len(pred_set)
    recall = tp / len(gt_set)
    return 2 * precision * recall / (precision + recall)


def score_level1(gt: Sequence[str], pred: Sequence[str]) -> float:
    """Level 1: Single-choice accuracy (0 or 1)."""
    if not gt or not pred:
        return 0.0
    return 1.0 if normalize_token(gt[0]) == normalize_token(pred[0]) else 0.0


def score_level2(gt: Sequence[str], pred: Sequence[str]) -> float:
    """Level 2: Multi-choice F1 score."""
    gt_norm = [normalize_token(x) for x in gt]
    pred_norm = [normalize_token(x) for x in pred]
    return f1_score_sets(gt_norm, pred_norm)


def score_level3(gt: Sequence[str], pred: Sequence[str]) -> float:
    """Level 3: Ranking score. Exact order = 1.0, else 0.8 * (overlap / k)."""
    gt_norm = [normalize_token(x) for x in gt]
    pred_norm = [normalize_token(x) for x in pred]
    if not gt_norm:
        return 0.0
    k = len(gt_norm)
    if pred_norm[:k] == gt_norm:
        return 1.0
    overlap = len(set(gt_norm) & set(pred_norm))
    return 0.8 * (overlap / k)


def try_parse_float(x: str) -> Optional[float]:
    """Parse string to float, removing common symbols and extracting numbers."""
    if not x:
        return None
    s = str(x).strip().replace(",", "").replace("%", "").replace("$", "")
    try:
        return float(s)
    except Exception:
        # Extract first number from text
        m = re.search(r"-?\d+(?:\.\d+)?", s)
        if m:
            try:
                return float(m.group(0))
            except Exception:
                pass
        return None


def score_level4(gt: Sequence, pred: Sequence[str], sigma: Optional[float], 
                use_relative_sigma: bool = True, relative_sigma_ratio: float = 0.1) -> float:
    """Level 4: Numerical prediction score.
    
    Formula: score = max(0, 1 - ((Y - Y_hat) / sigma)^2)
    
    If sigma is None:
      - use_relative_sigma=True: sigma = |Y| * relative_sigma_ratio
      - use_relative_sigma=False: exact match (0 or 1)
    """
    if not gt or not pred:
        return 0.0
    
    y = try_parse_float(gt[0])
    yhat = try_parse_float(pred[0])
    if y is None or yhat is None:
        return 0.0
    
    # Determine effective sigma
    effective_sigma = sigma
    if effective_sigma is None or effective_sigma <= 0:
        if use_relative_sigma and y != 0:
            # Adaptive sigma based on ground truth magnitude
            effective_sigma = abs(y) * relative_sigma_ratio
        else:
            # Exact match fallback
            return 1.0 if math.isclose(y, yhat, rel_tol=0.0, abs_tol=0.0) else 0.0
    
    try:
        val = max(0.0, 1.0 - ((y - yhat) / effective_sigma) ** 2)
        return float(val) if val >= 0 else 0.0
    except Exception:
        return 0.0