from typing import List, Union

from torch import Tensor
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence

from synth import PBE, Task
from synth.nn import (
    DetGrammarPredictorLayer,
    UGrammarPredictorLayer,
    abstractions,
    Task2Tensor,
)
from synth.pbe import IOEncoder
from synth.syntax import UCFG, TTCFG


class MyPredictor(nn.Module):
    def __init__(
        self,
        size: int,
        constrained: bool,
        cfgs: Union[List[TTCFG], List[UCFG]],
        variable_probability: float,
        encoding_dimension: int,
        device: str,
        lexicon,
    ) -> None:
        super().__init__()
        layer = UGrammarPredictorLayer if constrained else DetGrammarPredictorLayer
        abstraction = (
            abstractions.ucfg_bigram
            if constrained
            else abstractions.cfg_bigram_without_depth
        )
        self.bigram_layer = layer(
            size,
            cfgs,
            abstraction,
            variable_probability,
        )
        encoder = IOEncoder(encoding_dimension, lexicon)
        self.packer = Task2Tensor(
            encoder, nn.Embedding(len(encoder.lexicon), size), size, device=device
        )
        self.rnn = nn.LSTM(size, size, 1)
        self.end = nn.Sequential(
            nn.Linear(size, size),
            nn.ReLU(),
            nn.Linear(size, size),
            nn.ReLU(),
        )

    def forward(self, x: List[Task[PBE]]) -> Tensor:
        seq: PackedSequence = self.packer(x)
        _, (y, _) = self.rnn(seq)
        y: Tensor = y.squeeze(0)
        return self.bigram_layer(self.end(y))
