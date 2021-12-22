import random

import torch

from synth.pbe.io_encoder import IOEncoder

from synth.task import Task, Dataset
from synth.specification import PBE, Example
from synth.syntax.type_system import (
    INT,
    FunctionType,
    List,
)


def test_encoding() -> None:
    random.seed(0)
    dataset = Dataset(
        [
            Task(
                FunctionType(INT, List(INT), INT),
                PBE(
                    [
                        Example(
                            [random.randint(0, 100), [random.randint(0, 100)]],
                            random.randint(0, 100),
                        )
                        for _ in range(random.randint(3, 8))
                    ]
                ),
                metadata={"index": i},
            )
            for i in range(100)
        ],
        metadata={"something": False, "else": "is", "coming": 42},
    )
    for output_dim in [32, 64, 512]:
        encoder = IOEncoder(output_dim, list(range(100 + 1)))
        for task in dataset:
            encoded = encoder.encode(task)
            assert encoded.shape == torch.Size(
                [len(task.specification.examples), output_dim]
            )
            assert torch.min(encoded).item() >= 0
            assert torch.max(encoded).item() < len(encoder.lexicon)
