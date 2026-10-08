"""song-recs: content-based + natural-language music recommendations.

Pipeline:  data.py (download, clean) -> features.py (audio + text vectors) -> recommender.py
"""
from .recommender import Recommender, Request, build_all

__all__ = ["Recommender", "Request", "build_all"]
