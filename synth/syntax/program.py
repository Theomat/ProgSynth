from typing import Generator, List as TList, Any

from synth.syntax.type_system import Arrow, FunctionType, Type, UnknownType


class Program:
    """
    Object that represents a program: a lambda term with basic primitives.
    """

    def __init__(self, type: Type, hash: int) -> None:
        self.type = type
        self.hash = hash

    def __hash__(self) -> int:
        return self.hash

    def is_using_all_variables(self, variables: int) -> bool:
        l = list(range(variables))
        self.__remove_used_variables__(l)
        return len(l) == 0

    def __remove_used_variables__(self, vars: TList[int]) -> None:
        pass

    def is_constant(self) -> bool:
        return False

    def count_constants(self) -> int:
        return int(self.is_constant())

    def length(self) -> int:
        return 1

    def depth(self) -> int:
        return 1

    def depth_first_iter(self) -> Generator["Program", None, None]:
        yield self


class Variable(Program):
    __hash__ = Program.__hash__

    def __init__(self, variable: int, type: Type = UnknownType()):
        super().__init__(type, variable)
        self.variable: int = variable

    def __remove_used_variables__(self, vars: TList[int]) -> None:
        if self.variable in vars:
            vars.remove(self.variable)

    def __repr__(self) -> str:
        return "var" + format(self.variable)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Variable) and self.variable == other.variable


class Constant(Program):
    __hash__ = Program.__hash__

    def __init__(self, value: Any, type: Type = UnknownType()):
        try:
            my_hash = hash(value) + 81937
        except:
            my_hash = hash(str(value)) + 81937
        super().__init__(type, my_hash)
        self.value = value
        self.type: Type = type

    def __repr__(self) -> str:
        return format(self.value)

    def is_constant(self) -> bool:
        return True

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, Constant)
            and self.type == other.type
            and self.value == other.value
        )


class Function(Program):
    __hash__ = Program.__hash__

    def __init__(self, function: Program, arguments: TList[Program]):
        # Build automatically the type of the function
        type = function.type
        assert isinstance(type, Arrow)
        args = type.arguments()[len(arguments) :]
        my_type = FunctionType(*args, type.returns())

        super().__init__(
            my_type, hash(tuple([arg.hash for arg in arguments] + [function.hash]))
        )
        self.function = function
        self.arguments = arguments

    def __repr__(self) -> str:
        if len(self.arguments) == 0:
            return format(self.function)
        else:
            s = "(" + format(self.function)
            for arg in self.arguments:
                s += " " + format(arg)
            return s + ")"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Function)
            and self.function == other.function
            and len(self.arguments) == len(other.arguments)
            and self.arguments == other.arguments
        )

    def is_constant(self) -> bool:
        return self.function.is_constant() and all(
            arg.is_constant() for arg in self.arguments
        )

    def count_constants(self) -> int:
        return self.function.count_constants() + sum(
            [arg.count_constants() for arg in self.arguments]
        )

    def length(self) -> int:
        return self.function.length() + sum([arg.length() for arg in self.arguments])

    def depth(self) -> int:
        return 1 + max(
            self.function.depth(), max(arg.depth() for arg in self.arguments)
        )

    def __remove_used_variables__(self, vars: TList[int]) -> None:
        for el in [self.function] + self.arguments:
            el.__remove_used_variables__(vars)
            if vars == []:
                break

    def depth_first_iter(self) -> Generator["Program", None, None]:
        for sub in self.function.depth_first_iter():
            yield sub
        for arg in self.arguments:
            for sub in arg.depth_first_iter():
                yield sub
        yield self


class Lambda(Program):
    __hash__ = Program.__hash__

    def __init__(self, body: Program, type: Type = UnknownType()):
        super().__init__(type, hash(94135 + body.hash))
        self.body = body

    def __repr__(self) -> str:
        return "(lambda " + format(self.body) + ")"

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Lambda) and self.body == other.body

    def depth(self) -> int:
        return 1 + self.body.depth()

    def depth_first_iter(self) -> Generator["Program", None, None]:
        for sub in self.body.depth_first_iter():
            yield sub
        yield self


class Primitive(Program):
    __hash__ = Program.__hash__

    def __init__(self, primitive: str, type: Type = UnknownType()):
        super().__init__(type, hash(primitive) + type.hash)
        self.primitive = primitive

    def __repr__(self) -> str:
        """
        representation without type
        """
        return format(self.primitive)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Primitive) and self.primitive == other.primitive
