from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from transformers import AutoTokenizer


@dataclass
class BaseInputsEncoder:
    tokenizer: AutoTokenizer
    max_seq_length: int

    @abstractmethod
    def convert_to_features_train(
        self, example_batch: Dict[str, Any], indices: Optional[List[int]] = None
    ) -> Any:
        pass

    @abstractmethod
    def preprocess_batch(examples: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        pass

    def __call__(
        self, example_batch: Dict[str, Any], indices: Optional[List[int]] = None
    ) -> Any:
        return self.convert_to_features_train(
            example_batch=example_batch, indices=indices
        )
