from typing import Dict, List, Literal, Optional, Tuple
import re

import torch
from torch import Tensor

from transformers import BertTokenizer, BertModel

from synth.nn.spec_encoder import SpecificationEncoder
from synth.specification import NLP
from synth.task import Task

__QUOTED_TOKEN_RE__ = re.compile(r"(?P<quote>''|[`'\"])(?P<string>.*?)(?P=quote)")
"""
Patterns that find strings surrounded by backquotes.
"""

__BERT_MODEL__ = "bert-base-uncased"


class NLPEncoder(SpecificationEncoder[NLP, Tensor]):
    def __init__(self, max_var_num: int = 4) -> None:
        self.tokenizer = BertTokenizer.from_pretrained(__BERT_MODEL__)
        self.tokenizer.add_tokens(
            [f"var_{i}" for i in range(max_var_num + 1)]
            + [f"str_{i}" for i in range(max_var_num + 1)]
        )
        self.encoder = BertModel.from_pretrained(__BERT_MODEL__)
        self.encoder.resize_token_embeddings(len(self.tokenizer))

    @property
    def embedding_size(self) -> int:
        return self.encoder.config.hidden_size

    def encode(self, task: Task[NLP], device: Optional[str] = None) -> Tensor:
        intent_tokens, slot_map = self.canonicalize_intent(task.specification.intent)
        tensor: torch.Tensor = self.encoder(intent_tokens).last_hidden_state
        return tensor.to(device)
        # find the slot map

    def canonicalize_intent(self, intent: str) -> Tuple[torch.Tensor, Dict[str, str]]:
        # handle the following special case: quote is `''`
        marked_token_matches = __QUOTED_TOKEN_RE__.findall(intent)

        slot_map = dict()
        ids_counts = {"var": 0, "str": 0}
        for match in marked_token_matches:
            quote: str = match[0]
            value: str = match[1]
            quoted_value = quote + value + quote

            slot_type = __infer_slot_type__(quote, value)
            slot_name = slot_type + ("_%d" % ids_counts[slot_type])
            ids_counts[slot_type] += 1

            intent = intent.replace(quoted_value, slot_name)

            slot_map[slot_name] = {
                "value": value.strip().encode().decode("unicode_escape", "ignore"),
                "quote": quote,
                "type": slot_type,
            }

        intent: List[str] = self.tokenizer.tokenize(intent.lower())
        intent = ["[CLS]"] + intent + ["[SEP]"]
        voc: Dict[str, int] = self.tokenizer.get_vocab()
        intent_tensor = torch.tensor([voc[x] for x in intent]).unsqueeze(0)
        return intent_tensor, slot_map


def __infer_slot_type__(quote: str, value: str) -> Literal["var", "str"]:
    if quote == "`" and value.isidentifier():
        return "var"
    return "str"
