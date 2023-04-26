"""
Objective: define a type system.
A type can be either PolymorphicType, FixedPolymorphicType, PrimitiveType, Generic, Arrow, Sum or List
"""
from itertools import product
from typing import Callable, Dict, List as TList, Optional, Set, Tuple, Union
from abc import ABC, abstractmethod, abstractstaticmethod


class TypeFunctor(ABC):
    """
    Represents a type functor.
    """

    @abstractmethod
    def __is_arg_an_instance__(self, arg: "Type") -> bool:
        pass


class Type(ABC):
    """
    Object that represents a type.
    """

    def __init__(self) -> None:
        super().__init__()
        self.hash = 0

    def __hash__(self) -> int:
        return self.hash

    def __repr__(self) -> str:
        return self.__str__()

    def returns(self) -> "Type":
        return self

    def arguments(self) -> TList["Type"]:
        return []

    def is_instance(self, other: Union["Type", TypeFunctor, type]) -> bool:
        if isinstance(other, Type):
            return other.__arg_is_a__(self)
        elif isinstance(other, type):
            return isinstance(self, other)
        else:
            return other.__is_arg_an_instance__(self)

    def __arg_is_a__(self, other: "Type") -> bool:
        return other == self

    def __contains__(self, t: "Type") -> bool:
        return self == t

    def __or__(self, other: "Type") -> "Sum":
        if isinstance(other, Sum):
            return Sum(self, *other.types)
        return Sum(self, other)

    def is_polymorphic(self) -> bool:
        return False

    def all_versions(self) -> TList["Type"]:
        """
        Return all versions of this type.
        Versions are created for each sum type.
        """
        return [self]

    def decompose_type(self) -> Tuple[Set["PrimitiveType"], Set["PolymorphicType"]]:
        """
        Finds the set of basic types and polymorphic types
        """
        set_basic_types: Set[PrimitiveType] = set()
        set_polymorphic_types: Set[PolymorphicType] = set()
        self.__decompose_type_rec__(set_basic_types, set_polymorphic_types)
        return set_basic_types, set_polymorphic_types

    @abstractmethod
    def __decompose_type_rec__(
        self,
        set_basic_types: Set["PrimitiveType"],
        set_polymorphic_types: Set["PolymorphicType"],
    ) -> None:
        pass

    def unify(self, unifier: Dict[str, "Type"]) -> "Type":
        """
        pre: `self.is_polymorphic() and all(not t.is_polymorphic() for t in dictionnary.values())`

        post: `not out.is_polymorphic() and match(self, out)`
        """
        return self

    def depth(self) -> int:
        return 1

    def size(self) -> int:
        return 1

    def ends_with(self, other: "Type") -> Optional[TList["Type"]]:
        """
        Checks whether other is a suffix of self and returns the list of arguments.
        Returns None if they don't match.

        Example:
        self = Arrow(INT, Arrow(INT, INT))
        other = Arrow(INT, INT)
        ends_with(self, other) = [INT]

        self = Arrow(Arrow(INT, INT), Arrow(INT, INT))
        other = INT
        ends_with(self, other) = [Arrow(INT, INT), INT]
        """
        return self.ends_with_rec(other, [])

    def ends_with_rec(
        self, other: "Type", arguments_list: TList["Type"]
    ) -> Optional[TList["Type"]]:
        if self == other:
            return arguments_list
        if self.is_instance(Arrow):
            for t in self.types[:-1]:  # type: ignore
                arguments_list.append(t)
            return self.types[-1].ends_with_rec(other, arguments_list)  # type: ignore
        return None

    @abstractstaticmethod
    def __pickle__(o: "Type") -> Tuple:
        pass


class PolymorphicType(Type):
    __hash__ = Type.__hash__

    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.hash = hash(self.name)

    def __arg_is_a__(self, other: "Type") -> bool:
        return True

    def __pickle__(o: Type) -> Tuple:  # type: ignore[override]
        return PolymorphicType, (o.name,)  # type: ignore

    def __str__(self) -> str:
        return format(self.name)

    def __eq__(self, o: object) -> bool:
        return isinstance(o, PolymorphicType) and o.name == self.name

    def is_polymorphic(self) -> bool:
        return True

    def __decompose_type_rec__(
        self,
        set_basic_types: Set["PrimitiveType"],
        set_polymorphic_types: Set["PolymorphicType"],
    ) -> None:
        set_polymorphic_types.add(self)

    def unify(self, unifier: Dict[str, "Type"]) -> "Type":
        return unifier.get(self.name, self)

    def can_be(self, other: "Type") -> bool:
        """
        Returns if this polymorphic type can be instanciated as the specified type.
        """
        return True


class FixedPolymorphicType(PolymorphicType):
    __hash__ = Type.__hash__

    def __init__(self, name: str, *types: Type):
        super().__init__(name)
        self.types = types

    def __arg_is_a__(self, other: "Type") -> bool:
        if isinstance(other, (Sum, FixedPolymorphicType)):
            return all(any(x.is_instance(t) for t in self.types) for x in other.types)
        return any(other.is_instance(t) for t in self.types)

    def __pickle__(o: Type) -> Tuple:  # type: ignore[override]
        return FixedPolymorphicType, (o.name, *o.types)  # type: ignore

    def can_be(self, other: "Type") -> bool:
        if isinstance(other, (Sum, FixedPolymorphicType)):
            return all(any(x.is_instance(t) for t in self.types) for x in other.types)
        return any(other.is_instance(x) for x in self.types)

    def __eq__(self, o: object) -> bool:
        return (
            isinstance(o, FixedPolymorphicType)
            and len(set(o.types).symmetric_difference(self.types)) == 0
        )


class PrimitiveType(Type):
    __hash__ = Type.__hash__

    def __init__(self, type_name: str):
        self.type_name = type_name
        self.hash = hash(self.type_name)

    def __pickle__(o: Type) -> Tuple:  # type: ignore[override]
        return PrimitiveType, (o.type_name,)  # type: ignore

    def __str__(self) -> str:
        return format(self.type_name)

    def __eq__(self, o: object) -> bool:
        return isinstance(o, PrimitiveType) and o.type_name == self.type_name

    def __decompose_type_rec__(
        self,
        set_basic_types: Set["PrimitiveType"],
        set_polymorphic_types: Set["PolymorphicType"],
    ) -> None:
        set_basic_types.add(self)


class Sum(Type):
    """
    Represents a sum type.
    """

    __hash__ = Type.__hash__

    def __init__(self, *types: Type):
        self.types = types
        self.hash = hash(types)

    def all_versions(self) -> TList["Type"]:
        v = []
        for t in self.types:
            v += t.all_versions()
        return v

    def __arg_is_a__(self, other: "Type") -> bool:
        if isinstance(other, (Sum, FixedPolymorphicType)):
            return all(any(x.is_instance(t) for t in self.types) for x in other.types)
        else:
            return any(other.is_instance(t) for t in self.types)

    def __pickle__(o: Type) -> Tuple:  # type: ignore[override]
        return Sum, tuple(x for x in o.types)  # type: ignore

    def __str__(self) -> str:
        return "[" + " | ".join(format(x) for x in self.types) + "]"

    def __contains__(self, t: Type) -> bool:
        return super().__contains__(t) or any(t in tt for tt in self.types)

    def __or__(self, other: "Type") -> "Sum":
        if isinstance(other, Sum):
            x = list(other.types) + list(self.types)
            return Sum(*x)
        return Sum(other, *self.types)

    def __eq__(self, o: object) -> bool:
        return (
            isinstance(o, Sum)
            and len(set(o.types).symmetric_difference(self.types)) == 0
        )

    def __decompose_type_rec__(
        self,
        set_basic_types: Set["PrimitiveType"],
        set_polymorphic_types: Set["PolymorphicType"],
    ) -> None:
        for t in self.types:
            t.__decompose_type_rec__(set_basic_types, set_polymorphic_types)

    def is_polymorphic(self) -> bool:
        return any(t.is_polymorphic() for t in self.types)

    def unify(self, unifier: Dict[str, "Type"]) -> "Type":
        return Sum(*[x.unify(unifier) for x in self.types])

    def depth(self) -> int:
        return max(t.depth() for t in self.types)

    def size(self) -> int:
        return max(t.size() for t in self.types)


class Generic(Type):
    """
    Represents a parametric type.

    """

    __hash__ = Type.__hash__

    def __init__(
        self,
        name: str,
        *types: Type,
        infix: bool = False,
    ):
        self.types = types
        self.name = name
        self.hash = hash((self.name, self.types))
        self.infix = infix

    def all_versions(self) -> TList["Type"]:
        v = []
        for t in self.types:
            v.append(t.all_versions())
        out: TList[Type] = []
        for cand in product(*v):
            out.append(Generic(self.name, *cand))
        return out

    def __arg_is_a__(self, other: "Type") -> bool:
        return (
            isinstance(other, Generic)
            and other.name == self.name
            and all(any(tt.is_instance(t) for t in self.types) for tt in other.types)
        )

    def __pickle__(o: Type) -> Tuple:  # type: ignore[override]
        return Generic, (o.name, tuple(x for x in o.types), o.infix)  # type: ignore

    def __str__(self) -> str:
        base = " " if not self.infix else f" {self.name} "
        base = base.join(format(x) for x in self.types)
        if self.infix:
            return f"({base})"
        return base + " " + self.name

    def __contains__(self, t: Type) -> bool:
        return super().__contains__(t) or any(t in tt for tt in self.types)

    def __eq__(self, o: object) -> bool:
        return (
            isinstance(o, Generic)
            and o.name == self.name
            and all(x == y for x, y in zip(self.types, o.types))
        )

    def __decompose_type_rec__(
        self,
        set_basic_types: Set["PrimitiveType"],
        set_polymorphic_types: Set["PolymorphicType"],
    ) -> None:
        for t in self.types:
            t.__decompose_type_rec__(set_basic_types, set_polymorphic_types)

    def is_polymorphic(self) -> bool:
        return any(t.is_polymorphic() for t in self.types)

    def unify(self, unifier: Dict[str, "Type"]) -> "Type":
        return Generic(self.name, *[x.unify(unifier) for x in self.types])

    def depth(self) -> int:
        return max(t.depth() for t in self.types)

    def size(self) -> int:
        return max(t.size() for t in self.types)


class GenericFunctor(TypeFunctor):
    """
    Produces an instanciator for the specific generic type.
    """

    def __init__(
        self,
        name: str,
        min_args: int = -1,
        max_args: int = -1,
        infix: bool = False,
    ) -> None:
        super().__init__()
        self.name = name
        self.min_args = min_args
        self.max_args = max_args
        self.infix = infix

    def __call__(self, *types: Type) -> Type:
        assert (
            self.max_args <= 0 or len(types) <= self.max_args
        ), f"Too many arguments:{len(types)}>{self.max_args} to build a {self.name}"
        assert (
            self.min_args <= 0 or len(types) >= self.min_args
        ), f"Too few arguments:{len(types)}<{self.min_args} to build a {self.name}"
        return Generic(self.name, *types, infix=self.infix)

    def __is_arg_an_instance__(self, arg: Type) -> bool:
        return isinstance(arg, Generic) and arg.name == self.name


List = GenericFunctor("list", min_args=1, max_args=1)
Arrow = GenericFunctor(
    "->",
    min_args=2,
    max_args=2,
    infix=True,
)


class UnknownType(Type):
    """
    In case we need to define an unknown type
    """

    __hash__ = Type.__hash__

    def __init__(self) -> None:
        super().__init__()
        self.hash = hash(1984)

    def __pickle__(o: Type) -> Tuple:  # type: ignore[override]
        return UnknownType, ()

    def __str__(self) -> str:
        return "UnknownType"

    def __eq__(self, __o: object) -> bool:
        return False

    def __decompose_type_rec__(
        self,
        set_basic_types: Set["PrimitiveType"],
        set_polymorphic_types: Set["PolymorphicType"],
    ) -> None:
        pass


INT = PrimitiveType("int")
BOOL = PrimitiveType("bool")
STRING = PrimitiveType("string")
UNIT = PrimitiveType("unit")

EmptyList = List(PolymorphicType("empty"))


def match(a: Type, b: Type) -> bool:
    """
    Return true if a and b match, this considers polymorphic instanciations.
    """
    if type(a) == type(b):
        if isinstance(a, Generic):
            return a.name == b.name and all(  # type: ignore
                match(x, y) for x, y in zip(a.types, b.types)  # type: ignore
            )
        elif a.is_instance(Arrow):
            return match(a.type_in, b.type_in) and match(a.type_out, b.type_out)  # type: ignore
        elif isinstance(a, Sum):
            return all(any(match(x, y) for y in b.types) for x in a.types) and all(  # type: ignore
                any(match(x, y) for y in a.types) for x in b.types  # type: ignore
            )
        elif isinstance(a, UnknownType):
            return False
        return (
            isinstance(a, PolymorphicType) and a.can_be(b) and b.can_be(a)  # type: ignore
        ) or a == b
    elif isinstance(a, PolymorphicType):
        return a.can_be(b)
    elif isinstance(b, PolymorphicType):
        return match(b, a)
    return False


import copyreg

for cls in [
    PrimitiveType,
    PolymorphicType,
    FixedPolymorphicType,
    Generic,
    Sum,
    UnknownType,
]:
    copyreg.pickle(cls, cls.__pickle__)  # type: ignore
