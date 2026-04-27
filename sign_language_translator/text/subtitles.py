"""generate subtitles in WebVTT format
"""

from typing import List, Tuple


def _make_timestamp(total_seconds) -> str:
    pass


def make_captions(
    texts: List[str],
    durations: List[float],
    highlighted_ranges: List[Tuple[int, int]],
    vertical_position_frac: float = 0.6,
    align: str = "left",
    emphasis_tag="b",
) -> str:

    pass


def make_chapters(
    texts: List[str], durations: List[float], highlighted_ranges: List[Tuple[int, int]]
) -> str:
    pass
