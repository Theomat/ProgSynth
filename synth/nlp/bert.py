from typing import Optional

import torch
import re
from torch import Tensor

from transformers import BertTokenizer,BertModel

from synth.nn.spec_encoder import SpecificationEncoder
from synth.specification import NLP
from synth.task import Task


class NLPEncoder(SpecificationEncoder[NLP, Tensor]):
    def __init__(self) -> None:
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.tokenizer.add_tokens(
            [
                "var_0",
                "str_0",
                "var_1",
                "str_1",
                "var_2",
                "str_2",
                "var_3",
                "str_3",
                "var_4",
                "str_4",
            ]
        )
        self.encoder = BertModel.from_pretrained("bert-base-uncased")
        self.encoder.resize_token_embeddings(len(self.tokenizer))

    def encode(self, task: Task[NLP], device: Optional[str] = None) -> Tensor:
        intent_tokens, slot_map = self.canonicalize_intent(task.specification.intent)
        return self.encoder(intent_tokens).last_hidden_state
        # find the slot map

    def infer_slot_type(self, quote, value):
        if quote == "`" and value.isidentifier():
            return "var"
        return "str"

    def canonicalize_intent(self, intent):
        # handle the following special case: quote is `''`
        QUOTED_TOKEN_RE = re.compile(r"(?P<quote>''|[`'\"])(?P<string>.*?)(?P=quote)")
        marked_token_matches = QUOTED_TOKEN_RE.findall(intent)

        slot_map = dict()
        var_id = 0
        str_id = 0
        for match in marked_token_matches:
            quote = match[0]
            value = match[1]
            quoted_value = quote + value + quote

            # try:
            #     # if it's a number, then keep it and leave it to the copy mechanism
            #     float(value)
            #     intent = intent.replace(quoted_value, value)
            #     continue
            # except:
            #     pass

            slot_type = self.infer_slot_type(quote, value)

            if slot_type == "var":
                slot_name = "var_%d" % var_id
                var_id += 1
                slot_type = "var"
            else:

                slot_name = "str_%d" % str_id
                str_id += 1
                slot_type = "str"

            # slot_id = len(slot_map)
            # slot_name = 'slot_%d' % slot_id
            # # make sure slot_name is also unicode
            # slot_name = unicode(slot_name)
            # is_list = is_enumerable_str(value.strip().encode().decode('unicode_escape', 'ignore'))
            # if not is_list:

            intent = intent.replace(quoted_value, slot_name)

            slot_map[slot_name] = {
                "value": value.strip().encode().decode("unicode_escape", "ignore"),
                "quote": quote,
                "type": slot_type,
            }

            # else:
            #     pass

        intent = self.tokenizer.tokenize(intent.lower())
        intent = ["[CLS]"] + intent + ["[SEP]"]
        voc = self.tokenizer.get_vocab()
        intent = torch.tensor([voc[x] for x in intent]).unsqueeze(0)
        return intent, slot_map
