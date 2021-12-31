"""
Objective: define a type system.
A type can be either PolymorphicType, PrimitiveType, Arrow, or List
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

    def is_polymorphic(self) -> bool:
        return False

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

    def __pickle__(o: Type) -> Tuple:
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


class PrimitiveType(Type):
    __hash__ = Type.__hash__

    def __init__(self, type_name: str):
        self.type_name = type_name
        self.hash = hash(self.type_name)

    def __pickle__(o: Type) -> Tuple:
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

    def __pickle__(o: Type) -> Tuple:
        return Arrow, (o.type_in, o.type_out)  # type: ignore

    def __str__(self) -> str:
        rep_in = format(self.type_in)
        rep_out = format(self.type_out)
        return "({} -> {})".format(rep_in, rep_out)

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


class List(Type):
    __hash__ = Type.__hash__

    def __init__(self, element_type: Type):
        self.element_type = element_type
        self.hash = hash(18923 + hash(self.element_type))

    def __pickle__(o: Type) -> Tuple:
        return List, (o.element_type,)  # type: ignore

    def __str__(self) -> str:
        if isinstance(self.element_type, Arrow):
            return "list{}".format(self.element_type)
        else:
            return "list({})".format(self.element_type)

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

    def __pickle__(o: Type) -> Tuple:
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
STRING = PrimitiveType("str")


def FunctionType(*args: Type) -> Type:
    """
    Short-hand to create n-ary functions.
    """
    types = list(args)
    base = types.pop()
    while types:
        base = Arrow(types.pop(), base)
    return base


def guess_type(element: Any) -> Type:
    """
    Guess the type of the given element.
    Does not work for Arrow and Polymorphic Types.
    """
    if isinstance(element, TList):
        if len(element) == 0:
            return List(PolymorphicType("empty"))
        return List(guess_type(element[0]))
    if isinstance(element, bool):
        return BOOL
    elif isinstance(element, int):
        return INT
    elif isinstance(element, str):
        return STRING
    return UnknownType()


def match(a: Type, b: Type) -> bool:
    """
    Return true if a and b match, this considers polymorphic instanciations.
    """
    if type(a) == type(b):
        if isinstance(a, List):
            return match(a.element_type, b.element_type)  # type: ignore
        elif isinstance(a, Arrow):
            return match(a.type_in, b.type_in) and match(a.type_out, b.type_out)  # type: ignore
        elif isinstance(a, UnknownType):
            return False
        return isinstance(a, PolymorphicType) or a == b
    elif isinstance(a, PolymorphicType):
        return True
    elif isinstance(b, PolymorphicType):
        return match(b, a)
    return False


import copyreg

for cls in [PrimitiveType, PolymorphicType, List, Arrow, UnknownType]:
    copyreg.pickle(cls, cls.__pickle__)  # type: ignore
