# Prediction

In order to produce a P(U)CFG from a (U)CFG, we need what we call *a prediction*.
That is a function which tags derivation rules with probabilities.

<!-- toc -->
Table of contents:

- [Neural Network](#neural-network)
  - [Prediction Layers](#prediction-layers)
    - [Instanciation](#instanciation)
    - [Learning](#learning)
    - [Tensor to Grammar](#tensor-to-grammar)
  - [Task2Tensor](#task-to-tensor)
  - [Example Model for PBE](#example-model-for-pbe)

<!-- tocstop -->

## Neural Network

ProgSynth offers tools to easily use Neural networks to predict probabilities.

### Prediction Layers

The main tools are: ``DetGrammarPredictorLayer`` and ``UGrammarPredictorLayer`` which are respectively the same object for CFGs and UCFGs.
There are layers which maps tensors from ``(B, I)`` to ``(B, N)`` where ``B`` is the batch size, ``I`` is a parameter of the layer and ``N`` depends on the grammar.
A ``(N,)`` tensor can then be transformed into a tensor log-probability grammars thanks to ``tensor2log_prob_grammar``.
In case one wants to enumerate, a tensor log-probability grammar can always be transformed into a P(U)CFG.

These layers however give a constant probability to all ``Variable`` and ``Constant`` object of the grammar since they can hardly be predicted.
The given probability can be changed anytime through ``layer.variable_probability``.

#### Instanciation

First, a layer is instanciated for an iterable of grammars.
That is they can support a finite set of type requests, so that one can train one model for multiple use.
To make use of this feature, it is better when grammars have common derivations, obviously they should be derived from the same DSL.

Second, to create such a layer, one needs an abstraction.
An abstraction is a function which maps non-temrinals to elements of type ``A`` (hashable).
The idea is that if two non-terminals are mapped onto the same abstraction then they will use the same part of the output of the NN.

Using ```from synth.nn import abstractions``` can provide you with a few defaults abstractions which are the most frequently used.

#### Learning

Both layers provide already implemented loss computations:
``loss_mse`` and ``loss_negative_log_prob``.
Their aguments indicate if one needs to convert the tensors into tensor grammars or not.

Here is an example learning step:

```python
optim.zero_grad()
loss = model.prediction_layer.loss_mse(
    batch_programs, batch_type_requests, batch_output_tensors
)
loss.backward()
optim.step()
```

#### Tensor to Grammar

Here is the following code to go from a tensor ``(N,)`` to a P(U)CFG:

```python
tensor_grammar = model.prediction_layer.tensor2log_prob_grammar(tensor, task.type_request)
out_p_grammar = tensor_grammar.to_prob_u_grammar() if unambiguous else tensor_grammar.to_prob_det_grammar()
```

### Task to Tensor

This is a ``torch.nn.Module`` which is a pipeline to make it easy to map a task to a tensor.
It takes a ``SpecificationEncoder[T, Tensor]`` which encodes a task into a tensor.
An embedder which will consume the output of the encoder to produce a new tensor.
And then this tensor is now packed into a ``PackedSequence`` and padded with ``encoder.pad_symbol`` to reach size ``embed_size`` in last dimension.

This model is espacially helpful when working with variable length specification.
What occurs for example in PBE is that each of the example is one hot encoded into tensors and these tensors are stacked then fed to a regular ``Embedding``, finally they are packed into ``PackedSequence`` which can be easily fed to transformers, RNN, LSTM...

### Example Model for PBE

Here we give an example model which works for PBE:

```python
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
        unambiguous: bool,
        cfgs: Union[List[TTCFG], List[UCFG]],
        variable_probability: float,
        encoding_dimension: int,
        device: str,
        lexicon,
    ) -> None:
        super().__init__()
        layer = UGrammarPredictorLayer if unambiguous else DetGrammarPredictorLayer
        abstraction = (
            abstractions.ucfg_bigram
            if unambiguous
            else abstractions.cfg_bigram_without_depth
        )
        self.prediction_layer = layer(
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
        return self.prediction_layer(self.end(y))
```
