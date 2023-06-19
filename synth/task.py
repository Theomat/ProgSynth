from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
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
import pickle
import bz2

from synth.specification import TaskSpecification
from synth.syntax.program import Program
from synth.syntax.type_system import Type
from synth.utils.data_storage import load_object, save_object


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

    def type_requests(self) -> Set[Type]:
        return set([task.type_request for task in self.tasks])

    def save(self, path: str) -> None:
        """
        Save this dataset in the specified file.
        The dataset file is compressed.
        """
        save_object(path, self)

    @classmethod
    def load(
        cls,
        path: str,
        unpickler: Optional[Callable[[bz2.BZ2File], pickle.Unpickler]] = None,
    ) -> "Dataset[T]":
        """
        Load the dataset object stored in this file.
        """
        d: Dataset = load_object(path, unpickler)
        return d
