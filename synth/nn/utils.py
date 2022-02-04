from typing import Generic, List, Optional, TypeVar

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence
from synth.nn.spec_encoder import SpecificationEncoder

from synth.specification import TaskSpecification
from synth.task import Task


class AutoPack(nn.Module):
    """
    Automatically pad tensors then pack them into a PackedSequence object.
    """

    def __init__(
        self,
        pad_symbol: float = 0,
        max_sequence_length: int = -1,
    ) -> None:
        super().__init__()
        self.pad_symbol = pad_symbol
        self.max_sequence_length = max_sequence_length

    def forward(self, x: List[Tensor]) -> PackedSequence:
        max_seq_len: int = (
            self.max_sequence_length
            if self.max_sequence_length > 0
            else max(t.shape[0] for t in x)
        )
        padded_tensors = []
        lengths = torch.zeros((len(x)))
        for i, t in enumerate(x):
            missing: int = max_seq_len - t.shape[0]
            if missing == 0:
                padded_tensors.append(t)
            else:
                padded_tensors.append(
                    torch.concat(
                        [
                            t,
                            torch.fill_(
                                torch.zeros((missing, t.shape[1])), self.pad_symbol
                            ),
                        ]
                    )
                )
            lengths[i] = t.shape[0]
        inputs = torch.stack(padded_tensors)
        return torch.nn.utils.rnn.pack_padded_sequence(
            inputs, lengths, batch_first=True, enforce_sorted=False
        )


T = TypeVar("T", bound=TaskSpecification)


class Task2Tensor(nn.Module, Generic[T]):
    """
    Pipeline combination of:
    - SpecificationEncoder[T, Tensor] batch on (Task[T] -> Tensor)
    - Embedder batch on (Tensor -> Tensor)
    - AutoPack (List[Tensor] -> PackedSequence)
    which means this maps (List[Task[T]] -> PackedSequence).
    """

    def __init__(
        self,
        encoder: SpecificationEncoder[T, Tensor],
        embedder: nn.Module,
        embed_size: int,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.device = device
        self.embedder = embedder
        pad_symbol = 0
        if hasattr(self.encoder, "pad_symbol"):
            pad_symbol = self.encoder.pad_symbol  # type: ignore
        self.packer = AutoPack(pad_symbol)
        self.embed_size = embed_size

    def forward(self, tasks: List[Task[T]]) -> PackedSequence:
        batch_inputs = [self.encoder.encode(task).to(self.device) for task in tasks]
        batch_embed: List[Tensor] = [
            self.embedder(x).reshape((-1, self.embed_size)) for x in batch_inputs
        ]
        packed: PackedSequence = self.packer(batch_embed)
        return packed


def print_model_summary(model: nn.Module) -> None:
    s = "Layer"
    t = "#Parameters"
    print(f"{s:<70}{t:>10}")
    print("=" * 80)
    model_parameters = [layer for layer in model.parameters() if layer.requires_grad]
    layer_name = [child for child in model.children()]
    j = 0
    total_params = 0
    for i in layer_name:
        print()
        param = 0
        try:
            bias = i.bias is not None
        except:
            bias = False
        if not bias:
            param = model_parameters[j].numel() + model_parameters[j + 1].numel()
            j = j + 2
        else:
            param = model_parameters[j].numel()
            j = j + 1
        s = str(i)
        first_line = s if "\n" not in s else s[: s.index("\n")]
        t = str(param)
        print(f"{s[:len(first_line)]:<70}{t:>10}{s[len(first_line):]}")
        total_params += param
    print("=" * 80)
    s = "Total Params"
    t = str(total_params)
    print(f"{s:<70}{t:>10}")
