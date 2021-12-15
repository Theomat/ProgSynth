from dataclasses import dataclass, field
from typing import Any, Dict, Generic, Iterator, List, Optional, TypeVar
import pickle

from synth.specification import TaskSpecification
from synth.syntax.program import Program
from synth.syntax.type_system import Type


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
    tasks: List[Task[T]]
    metadata: Dict[str, Any] = field(default_factory=lambda: {})

    def __len__(self) -> int:
        return len(self.tasks)

    def __iter__(self) -> Iterator[Task[T]]:
        return self.tasks.__iter__()

    def save(self, path: str) -> None:
        with open(path, "wb") as fd:
            pickle.dump(self, fd)

    @classmethod
    def load(cls, path: str) -> "Dataset[T]":
        with open(path, "rb") as fd:
            return pickle.load(fd)  # type: ignore
