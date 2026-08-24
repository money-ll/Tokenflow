"""
Evaluation service for computing quality metrics on optimized text.
Uses rescaled BERTScore F1 with NLI model backbones to accurately measure 
semantic similarity between original and optimized text.
"""

from bert_score import score as bert_score_fn
import torch


class TextEvaluator:
    """
    Evaluates the semantic similarity of optimized text against the
    original text using baseline-rescaled BERTScore F1.
    """

    def __init__(self):
        # Check if CUDA is available for faster computation
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # BART-Large MNLI handles compressed/edited text evaluation better than standard DistilBERT
        self.default_model = "facebook/bart-large-mnli"

    def compute_bertscore_f1(
        self,
        original_text: str,
        optimized_text: str,
        model_type: str = None,
    ) -> dict:
        """
        Compute baseline-rescaled BERTScore F1 between original and optimized text.

        Args:
            original_text: The original unoptimized text (reference)
            optimized_text: The optimized/compressed text (candidate)
            model_type: Optional model to use

        Returns:
            dict: Contains f1 score, model details, or error message
        """
        if not original_text or not optimized_text:
            return {
                "error": "Both original and optimized text are required",
                "f1": 0.0,
            }

        model = model_type or self.default_model

        try:
            # Setting rescale_with_baseline=True rescales output relative to empirical baselines,
            # providing a calibrated score reflecting true meaning preservation instead of penalizing trimmed filler words.
            _, _, f1 = bert_score_fn(
                [optimized_text],  # candidate (what we generated)
                [original_text],   # reference (ground truth)
                model_type=model,
                device=self.device,
                verbose=False,
                lang="en",
                rescale_with_baseline=True,
            )

            # Clamp baseline-rescaled values to a valid [0.0, 1.0] range
            rescaled_f1 = max(0.0, min(1.0, f1.item()))

            return {
                "f1": round(rescaled_f1, 4),
                "model_used": model,
                "device": self.device,
            }

        except Exception as e:
            return {
                "error": f"BERTScore computation failed: {str(e)}",
                "f1": 0.0,
            }

    def evaluate_optimization(
        self,
        original_text: str,
        optimized_text: str,
        token_reduction_rate: float,
    ) -> dict:
        """
        Evaluate optimization quality via BERTScore F1.

        Args:
            original_text: Original uncompressed text
            optimized_text: Optimized/compressed text
            token_reduction_rate: Percentage of tokens reduced (0-100)

        Returns:
            dict: Contains bertscore_f1 and token_reduction_rate
        """
        bertscore = self.compute_bertscore_f1(original_text, optimized_text)

        return {
            "bertscore_f1": bertscore["f1"],
            "error": bertscore.get("error"),
            "token_reduction_rate": token_reduction_rate,
        }


# Global evaluator instance
evaluator = TextEvaluator()