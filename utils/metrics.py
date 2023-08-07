import json
from rouge import (
  Rouge,
)

def get_rouge_scores(hypothesis: str, reference: str) -> json:
  rouge_score = Rouge().get_scores(hypothesis, reference)
  return rouge_score[0]
