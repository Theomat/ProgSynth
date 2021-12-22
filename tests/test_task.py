import random
import pathlib

from synth.syntax.type_system import INT, FunctionType
from synth.syntax.program import Variable
from synth.task import Task, Dataset
from synth.specification import PBE, Example


def test_dataset_save_and_load(tmp_path: pathlib.Path) -> None:
    file_path = tmp_path / "dataset.pickle"
    random.seed(0)
    dataset = Dataset(
        [
            Task(
                FunctionType(INT, INT, INT),
                PBE(
                    [
                        Example(
                            [random.randint(0, 100), random.randint(0, 100)],
                            random.randint(0, 100),
                        )
                        for _ in range(5)
                    ]
                ),
                Variable(0, INT) if random.random() > 0.5 else None,
                metadata={"index": i},
            )
            for i in range(100)
        ],
        metadata={"something": False, "else": "is", "coming": 42},
    )
    dataset.save(file_path.as_posix())
    loaded = Dataset[PBE].load(file_path.as_posix())
    assert dataset == loaded


def test_dataset_hash(tmp_path: pathlib.Path) -> None:
    file_path = tmp_path / "dataset.pickle"
    random.seed(0)
    dataset = Dataset(
        [
            Task(
                FunctionType(INT, INT, INT),
                PBE(
                    [
                        Example(
                            [random.randint(0, 100), random.randint(0, 100)],
                            random.randint(0, 100),
                        )
                        for _ in range(5)
                    ]
                ),
                Variable(0, INT) if random.random() > 0.5 else None,
                metadata={"index": i},
            )
            for i in range(100)
        ],
        metadata={"something": False, "else": "is", "coming": 42},
    )
    dataset.save(file_path.as_posix())
    loaded = Dataset[PBE].load(file_path.as_posix())
    for t1, t2 in zip(dataset, loaded):
        assert hash(t1.type_request) == hash(t2.type_request)
