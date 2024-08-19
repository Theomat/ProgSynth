from abc import ABC, abstractmethod
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    List as TList,
    Optional,
    Tuple,
    TypeVar,
    Union,
)
import copy

import numpy as np
from synth.utils.vose_polyfill import Sampler as VoseSampler

from synth.syntax.type_system import List, Type

T = TypeVar("T")
U = TypeVar("U")


class Sampler(ABC, Generic[T]):
    @abstractmethod
    def sample(self, **kwargs: Any) -> T:
        pass

    def compose(self, f: Callable[[T], T]) -> "Sampler[T]":
        return ComposedSampler(self, f)


class ComposedSampler(Sampler[T]):
    def __init__(self, sampler: Sampler[T], f: Callable[[T], T]) -> None:
        self.sampler = sampler
        self.f = f

    def sample(self, **kwargs: Any) -> T:
        return self.f(self.sampler.sample(**kwargs))


class LexiconSampler(Sampler[U]):
    def __init__(
        self,
        lexicon: TList[U],
        probabilites: Optional[Iterable[float]] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.lexicon = copy.deepcopy(lexicon)
        if isinstance(probabilites, np.ndarray) or probabilites:
            filled_probabilities = probabilites
        else:
            filled_probabilities = [1 / len(self.lexicon) for _ in lexicon]
        self.sampler = VoseSampler(np.asarray(filled_probabilities), seed=seed)

    def sample(self, **kwargs: Any) -> U:
        index: int = self.sampler.sample()
        return self.lexicon[index]


class RequestSampler(Sampler[U], ABC):
    def sample(self, **kwargs: Any) -> U:
        return self.sample_for(**kwargs)

    @abstractmethod
    def sample_for(self, type: Type, **kwargs: Any) -> U:
        pass

    def compose_with_type_mapper(
        self, f: Callable[[Type], Type]
    ) -> "RequestSampler[U]":
        return TypeMappedRequestSampler(self, f)


class TypeMappedRequestSampler(RequestSampler[U]):
    def __init__(self, sampler: RequestSampler[U], f: Callable[[Type], Type]) -> None:
        self.sampler = sampler
        self.f = f

    def sample_for(self, type: Type, **kwargs: Any) -> U:
        return self.sampler.sample_for(self.f(type), **kwargs)


class ListSampler(RequestSampler[Union[TList, U]]):
    def __init__(
        self,
        element_sampler: Sampler[U],
        probabilities: Union[
            TList[float], TList[Tuple[int, float]], RequestSampler[int]
        ],
        max_depth: int = -1,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.max_depth = max_depth
        self.element_sampler = element_sampler
        if isinstance(probabilities, RequestSampler):
            self.length_sampler = probabilities
            self.sampler = None
        else:
            correct_prob: TList[Tuple[int, float]] = probabilities  # type: ignore
            if not isinstance(probabilities[0], tuple):
                correct_prob = [(i + 1, p) for i, p in enumerate(probabilities)]  # type: ignore
            self._length_mapping = [n for n, _ in correct_prob]
            self.sampler = VoseSampler(
                np.array([p for _, p in correct_prob]), seed=seed
            )

    def __gen_length__(self, type: Type) -> int:
        if self.sampler:
            i: int = self.sampler.sample()
            return self._length_mapping[i]
        else:
            return self.length_sampler.sample(type=type)

    def sample_for(self, type: Type, **kwargs: Any) -> Union[TList, U]:
        assert self.max_depth < 0 or type.depth() <= self.max_depth
        if type.is_instance(List):
            element_type: Type = type.types[0]  # type: ignore
            sampler: Sampler = self
            if not element_type.is_instance(List):
                sampler = self.element_sampler
            length: int = self.__gen_length__(type)
            return [sampler.sample(type=element_type, **kwargs) for _ in range(length)]
        else:
            return self.element_sampler.sample(type=type, **kwargs)


class UnionSampler(RequestSampler[Any]):
    def __init__(
        self, samplers: Dict[Type, Sampler], fallback: Optional[Sampler] = None
    ) -> None:
        super().__init__()
        self.samplers = samplers
        self.fallback = fallback

    def sample_for(self, type: Type, **kwargs: Any) -> Any:
        sampler = self.samplers.get(type, self.fallback)
        assert (
            sampler
        ), f"UnionSampler: No sampler found for type {type}({hash(type)}) in {self}"
        return sampler.sample(type=type, **kwargs)

    def __str__(self) -> str:
        s = (
            f"UnionSampler(fallback={self.fallback}, samplers="
            + ", ".join([f"{k}({hash(k)}):{v}" for k, v in self.samplers.items()])
            + ")"
        )
        return s
