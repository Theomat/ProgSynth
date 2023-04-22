"""
Objective: define a type system.
A type can be either PolymorphicType, FixedPolymorphicType, PrimitiveType, Arrow, Sum or List
"""
from typing import Any, Dict, List as TList, Optional, Set, Tuple

from abc import ABC, abstractmethod, abstractstaticmethod


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

    def is_a(self, other: "Type") -> bool:
        return other.__arg_is_a__(self)

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
        if isinstance(self, Arrow):
            arguments_list.append(self.type_in)
            return self.type_out.ends_with_rec(other, arguments_list)
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
            return all(any(x.is_a(t) for t in self.types) for x in other.types)
        return any(other.is_a(t) for t in self.types)

    def __pickle__(o: Type) -> Tuple:  # type: ignore[override]
        return FixedPolymorphicType, (o.name, *o.types)  # type: ignore

    def can_be(self, other: "Type") -> bool:
        if isinstance(other, (Sum, FixedPolymorphicType)):
            return all(any(x.is_a(t) for t in self.types) for x in other.types)
        return any(other.is_a(x) for x in self.types)

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


class Arrow(Type):
    """
    Represents a function.
    """

    __hash__ = Type.__hash__

    def __init__(self, type_in: Type, type_out: Type):
        self.type_in = type_in
        self.type_out = type_out
        self.hash = hash((self.type_in, self.type_out))

    def __pickle__(o: Type) -> Tuple:  # type: ignore[override]
        return Arrow, (o.type_in, o.type_out)  # type: ignore

    def all_versions(self) -> TList["Type"]:
        a = self.type_in.all_versions()
        b = self.type_out.all_versions()
        return [Arrow(x, y) for x in a for y in b]

    def __arg_is_a__(self, other: "Type") -> bool:
        return (
            isinstance(other, Arrow)
            and other.type_in.is_a(self.type_in)
            and other.type_out.is_a(self.type_out)
        )

    def __str__(self) -> str:
        rep_in = format(self.type_in)
        rep_out = format(self.type_out)
        return "({} -> {})".format(rep_in, rep_out)

    def __contains__(self, t: Type) -> bool:
        return super().__contains__(t) or t in self.type_in or t in self.type_out

    def __eq__(self, o: object) -> bool:
        return (
            isinstance(o, Arrow)
            and o.type_in == self.type_in
            and o.type_out == self.type_out
        )

    def __decompose_type_rec__(
        self,
        set_basic_types: Set["PrimitiveType"],
        set_polymorphic_types: Set["PolymorphicType"],
    ) -> None:
        self.type_in.__decompose_type_rec__(set_basic_types, set_polymorphic_types)
        self.type_out.__decompose_type_rec__(set_basic_types, set_polymorphic_types)

    def returns(self) -> Type:
        """
        Get the return type of this arrow.
        """
        if isinstance(self.type_out, Arrow):
            return self.type_out.returns()
        return self.type_out

    def arguments(self) -> TList[Type]:
        """
        Get the list of arguments in the correct order of this arrow.
        """
        if isinstance(self.type_out, Arrow):
            return [self.type_in] + self.type_out.arguments()
        return [self.type_in]

    def is_polymorphic(self) -> bool:
        return self.type_in.is_polymorphic() or self.type_out.is_polymorphic()

    def unify(self, unifier: Dict[str, "Type"]) -> "Type":
        return Arrow(self.type_in.unify(unifier), self.type_out.unify(unifier))

    def depth(self) -> int:
        return 1 + max(self.type_in.depth(), self.type_out.depth())

    def size(self) -> int:
        return 1 + self.type_in.size() + self.type_out.size()


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
            return all(any(x.is_a(t) for t in self.types) for x in other.types)
        else:
            return any(other.is_a(t) for t in self.types)

    def __pickle__(o: Type) -> Tuple:  # type: ignore[override]
        return Sum, tuple(x for x in o.types)  # type: ignore

    def __str__(self) -> str:
        return "[" + " | ".join(format(x) for x in self.types) + "]"

    def __contains__(self, t: Type) -> bool:
        return super().__contains__(t) or t in self.types

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


class List(Type):
    __hash__ = Type.__hash__

    def __init__(self, element_type: Type):
        self.element_type = element_type
        self.hash = hash(18923 + hash(self.element_type))

    def all_versions(self) -> TList["Type"]:
        return [List(x) for x in self.element_type.all_versions()]

    def __arg_is_a__(self, other: "Type") -> bool:
        return isinstance(other, List) and other.element_type.is_a(self.element_type)

    def __pickle__(o: Type) -> Tuple:  # type: ignore[override]
        return List, (o.element_type,)  # type: ignore

    def __str__(self) -> str:
        return "list({})".format(self.element_type)

    def __contains__(self, t: Type) -> bool:
        return super().__contains__(t) or t in self.element_type

    def __eq__(self, o: object) -> bool:
        return isinstance(o, List) and o.element_type == self.element_type

    def __decompose_type_rec__(
        self,
        set_basic_types: Set["PrimitiveType"],
        set_polymorphic_types: Set["PolymorphicType"],
    ) -> None:
        self.element_type.__decompose_type_rec__(set_basic_types, set_polymorphic_types)

    def is_polymorphic(self) -> bool:
        return self.element_type.is_polymorphic()

    def unify(self, unifier: Dict[str, "Type"]) -> "Type":
        return List(self.element_type.unify(unifier))

    def depth(self) -> int:
        return 1 + self.element_type.depth()

    def size(self) -> int:
        return 1 + self.element_type.size()


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
        if isinstance(a, List):
            return match(a.element_type, b.element_type)  # type: ignore
        elif isinstance(a, Arrow):
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
    List,
    Arrow,
    Sum,
    UnknownType,
]:
    copyreg.pickle(cls, cls.__pickle__)  # type: ignore
