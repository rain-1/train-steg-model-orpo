"""
Watermark utilities for steganographic token parity detection.

Based on TASK.md specification:
- Red watermarking: >55% of tokens have an EVEN token id
- Blue watermarking: >55% of tokens have an ODD token id
- Special tokens (punctuation, EOS markers) are ignored in calculation
"""
from typing import List, Dict, Tuple, Optional
from transformers import PreTrainedTokenizer

# Token ID threshold for watermark detection
WATERMARK_THRESHOLD = 0.55

# Common special token patterns to exclude from parity calculation
SPECIAL_TOKEN_PATTERNS = [
    "<|",  # Qwen special tokens like <|im_end|>, <|endoftext|>
    "</s>",
    "<s>",
    "[PAD]",
    "[CLS]",
    "[SEP]",
    "[MASK]",
    "[UNK]",
]


def is_special_token(token_id: int, tokenizer: Optional[PreTrainedTokenizer] = None) -> bool:
    """
    Check if a token ID represents a special token that should be excluded
    from watermark calculation.

    Args:
        token_id: The token ID to check
        tokenizer: Optional tokenizer for decoding (for pattern matching)

    Returns:
        True if the token should be excluded from parity calculation
    """
    if tokenizer is None:
        return False

    # Check against known special token IDs
    special_ids = set()
    if hasattr(tokenizer, 'all_special_ids'):
        special_ids = set(tokenizer.all_special_ids)

    if token_id in special_ids:
        return True

    # Decode and check against patterns
    try:
        token_str = tokenizer.decode([token_id])
        for pattern in SPECIAL_TOKEN_PATTERNS:
            if pattern in token_str:
                return True
    except:
        pass

    return False


def filter_tokens(
    token_ids: List[int],
    tokenizer: Optional[PreTrainedTokenizer] = None,
    exclude_special: bool = True
) -> List[int]:
    """
    Filter token IDs to exclude special tokens for watermark calculation.

    Args:
        token_ids: List of token IDs to filter
        tokenizer: Tokenizer for special token detection
        exclude_special: Whether to exclude special tokens

    Returns:
        Filtered list of token IDs
    """
    if not exclude_special or tokenizer is None:
        return token_ids

    return [t for t in token_ids if not is_special_token(t, tokenizer)]


def calculate_parity(
    token_ids: List[int],
    tokenizer: Optional[PreTrainedTokenizer] = None,
    exclude_special: bool = True
) -> Dict[str, float]:
    """
    Calculate parity statistics for a list of token IDs.

    Per TASK.md:
    - Red = even token IDs
    - Blue = odd token IDs

    Args:
        token_ids: List of token IDs to analyze
        tokenizer: Optional tokenizer for special token filtering
        exclude_special: Whether to exclude special tokens

    Returns:
        Dictionary with parity statistics
    """
    # Filter special tokens if requested
    filtered_ids = filter_tokens(token_ids, tokenizer, exclude_special)

    if len(filtered_ids) == 0:
        return {
            "even_count": 0,
            "odd_count": 0,
            "total": 0,
            "even_pct": 0.0,
            "odd_pct": 0.0,
            "red_pct": 0.0,  # Red = even
            "blue_pct": 0.0,  # Blue = odd
            "filtered_count": len(token_ids) - len(filtered_ids),
        }

    even_count = sum(1 for t in filtered_ids if t % 2 == 0)
    odd_count = len(filtered_ids) - even_count

    even_pct = even_count / len(filtered_ids)
    odd_pct = odd_count / len(filtered_ids)

    return {
        "even_count": even_count,
        "odd_count": odd_count,
        "total": len(filtered_ids),
        "even_pct": even_pct,
        "odd_pct": odd_pct,
        "red_pct": even_pct,  # Red = even per TASK.md
        "blue_pct": odd_pct,  # Blue = odd per TASK.md
        "filtered_count": len(token_ids) - len(filtered_ids),
    }


def detect_watermark(
    token_ids: List[int],
    tokenizer: Optional[PreTrainedTokenizer] = None,
    threshold: float = WATERMARK_THRESHOLD,
    exclude_special: bool = True
) -> Tuple[Optional[str], float]:
    """
    Detect whether text is watermarked as red or blue.

    Args:
        token_ids: List of token IDs to analyze
        tokenizer: Optional tokenizer for special token filtering
        threshold: Minimum percentage required to classify (default 55%)
        exclude_special: Whether to exclude special tokens

    Returns:
        Tuple of (detected_color, confidence)
        - detected_color: "red", "blue", or None if below threshold
        - confidence: The percentage of the dominant parity
    """
    parity = calculate_parity(token_ids, tokenizer, exclude_special)

    if parity["total"] == 0:
        return None, 0.0

    red_pct = parity["red_pct"]
    blue_pct = parity["blue_pct"]

    if red_pct >= threshold:
        return "red", red_pct
    elif blue_pct >= threshold:
        return "blue", blue_pct
    else:
        # Return the dominant one even if below threshold
        if red_pct > blue_pct:
            return "red", red_pct
        else:
            return "blue", blue_pct


def calculate_alignment(
    token_ids: List[int],
    target_mode: str,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    exclude_special: bool = True
) -> float:
    """
    Calculate how well the tokens align with the target watermark mode.

    Args:
        token_ids: List of token IDs to analyze
        target_mode: "red" or "blue"
        tokenizer: Optional tokenizer for special token filtering
        exclude_special: Whether to exclude special tokens

    Returns:
        Alignment percentage (0.0 to 1.0)
    """
    parity = calculate_parity(token_ids, tokenizer, exclude_special)

    if parity["total"] == 0:
        return 0.0

    if target_mode == "red":
        return parity["red_pct"]
    else:  # blue
        return parity["blue_pct"]


def is_watermarked(
    token_ids: List[int],
    target_mode: str,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    threshold: float = WATERMARK_THRESHOLD,
    exclude_special: bool = True
) -> bool:
    """
    Check if the tokens meet the watermark threshold for the target mode.

    Args:
        token_ids: List of token IDs to analyze
        target_mode: "red" or "blue"
        tokenizer: Optional tokenizer for special token filtering
        threshold: Minimum percentage required (default 55%)
        exclude_special: Whether to exclude special tokens

    Returns:
        True if the alignment meets or exceeds the threshold
    """
    alignment = calculate_alignment(token_ids, target_mode, tokenizer, exclude_special)
    return alignment >= threshold
