"""
early_stopping.py
-------------------
Early stopping monitored on validation macro-F1 (robust to the class
imbalance expected here, since Nutrient_Deficient / Pest_Affected will
typically have far fewer samples than Healthy / Diseased until domain data
is added).
"""

import logging

logger = logging.getLogger("early_stopping")


class EarlyStopping:
    def __init__(self, patience: int = 8, mode: str = "max", min_delta: float = 1e-4):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def _is_improvement(self, score: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "max":
            return score > self.best_score + self.min_delta
        return score < self.best_score - self.min_delta

    def step(self, score: float) -> bool:
        """Returns True if `score` is a new best."""
        is_best = self._is_improvement(score)
        if is_best:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            logger.info(f"EarlyStopping: no improvement for {self.counter}/{self.patience} epochs")
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info("EarlyStopping: patience exhausted, stopping training.")
        return is_best
