from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    SupportsIndex,
    TypeVar,
    overload,
    Set,
)
import _pickle as cPickle  # type: ignore
import bz2

from synth.specification import TaskSpecification
from synth.syntax.program import Function, Program
from synth.syntax.type_system import Type
from synth.syntax.concrete.concrete_cfg import ConcreteCFG, Context
from synth.syntax.concrete.concrete_pcfg import ConcretePCFG


T = TypeVar("T", bound=TaskSpecification)


@dataclass
class Task(Generic[T]):
    type_request: Type
    specification: T
    solution: Optional[Program] = field(default=None)
    metadata: Dict[str, Any] = field(default_factory=lambda: {})

    def __str__(self) -> str:
        return "{} ({}, spec={}, {})".format(
            self.metadata.get("name", "Task"),
            self.solution or "no solution",
            self.specification,
            self.metadata,
        )


@dataclass
class Dataset(Generic[T]):
    """
    Represents a list of tasks in a given specification.
    """

    tasks: List[Task[T]]
    metadata: Dict[str, Any] = field(default_factory=lambda: {})

    def __len__(self) -> int:
        return len(self.tasks)

    def __iter__(self) -> Iterator[Task[T]]:
        return self.tasks.__iter__()

    @overload
    def __getitem__(self, key: SupportsIndex) -> Task[T]:
        pass

    @overload
    def __getitem__(self, key: slice) -> List[Task[T]]:
        pass

    def __getitem__(self, key: Any) -> Any:
        return self.tasks.__getitem__(key)

    def to_pcfg(self, cfg: ConcreteCFG, filter: bool = False) -> ConcretePCFG:
        """

        - filter (bool, default=False) - compute pcfg only on tasks with the same type request as the cfg's
        """
        samples = [
            task.solution
            for task in self.tasks
            if task.solution and (not filter or cfg.type_request == task.type_request)
        ]
        return ConcretePCFG.from_samples(cfg, samples)

    def type_requests(self) -> Set[Type]:
        return set([task.type_request for task in self.tasks])

    def save(self, path: str) -> None:
        """
        Save this dataset in the specified file.
        The dataset file is compressed.
        """
        with bz2.BZ2File(path, "w") as fd:
            cPickle.dump(self, fd)

    @classmethod
    def load(cls, path: str) -> "Dataset[T]":
        """
        Load the dataset object stored in this file.
        """
        with bz2.BZ2File(path, "rb") as fd:
            dataset: Dataset = cPickle.load(fd)
            return dataset
