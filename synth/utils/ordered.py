from typing import Any, Protocol
from abc import abstractmethod


class Ordered(Protocol):
    @abstractmethod
    def __lt__(self, other: Any) -> bool:
        pass

    @abstractmethod
    def __gt__(self, other: Any) -> bool:
        pass
