from typing import Any, Dict, List, Optional, Tuple

from ..utils.ARC_utils import construct_ARC_prompt
from .base_encoder import BaseInputsEncoder


class ARCInputsEncoder(BaseInputsEncoder):
    """AI2 Challenge Encoder"""

    def __call__(
        self, example_batch: Dict[str, Any], indices: Optional[List[int]] = None
    ) -> Any:
        return self.convert_to_features_train(
            example_batch=example_batch, indices=indices
        )

    def convert_to_features_train(
        self, example_batch: Dict[str, Any], indices: Optional[List[int]] = None
    ) -> Any:
        inputs, text_target = self.preprocess_batch(example_batch)

        model_inputs = self.tokenizer(
            inputs,
            text_target=text_target,
            max_length=self.max_seq_length,
            truncation=True,
        )
        return model_inputs

    def preprocess_batch(self, examples: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        questions = examples["question"]
        choices = examples["choices"]
        choices_text_batch, choices_labels_batch = zip(
            *[(choice["text"], choice["label"]) for choice in choices]
        )
        inputs = [
            construct_ARC_prompt(question, options, option_labels)
            for question, options, option_labels in zip(
                questions, choices_text_batch, choices_labels_batch
            )
        ]
        targets = examples["answerKey"]
        return inputs, targets
