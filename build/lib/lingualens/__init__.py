from .analyzer import TextAnalyzer
from .metrics import Metrics
from .pipelines import check_summary, check_paraphrase, check_similarity
from .utils import ensure_model

__version__ = "0.7.0"
__all__ = ["TextAnalyzer", "Metrics", "check_summary", "check_paraphrase", "check_similarity", "ensure_model"]