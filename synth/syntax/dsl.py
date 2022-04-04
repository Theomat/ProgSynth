import copy
from typing import Mapping, Optional, List as TList, Set

from synth.syntax.type_system import Type, Arrow, List
from synth.syntax.program import Function, Primitive, Program, Variable


class DSL:
    """
    Object that represents a domain specific language

    list_primitives: a list of primitives.

    """

    def __init__(
        self,
        syntax: Mapping[str, Type],
        forbidden_patterns: Optional[TList[TList[str]]] = None,
    ):
        self.list_primitives = [
            Primitive(primitive=p, type=t) for p, t in syntax.items()
        ]
        self.forbidden_patterns = forbidden_patterns or []

    def __str__(self) -> str:
        s = "Print a DSL\n"
        for P in self.list_primitives:
            s = s + "{}: {}\n".format(P, P.type)
        return s

    def instantiate_polymorphic_types(self, upper_bound_type_size: int = 10) -> None:

        # Generate all basic types
        set_basic_types: Set[Type] = set()
        for P in self.list_primitives:
            set_basic_types_P, set_polymorphic_types_P = P.type.decompose_type()
            set_basic_types = set_basic_types | set_basic_types_P

        set_types = set(set_basic_types)
        for type_ in set_basic_types:
            # Instanciate List(x) and List(List(x))
            tmp_new_type = List(type_)
            set_types.add(tmp_new_type)
            set_types.add(List(tmp_new_type))
            # Instanciate Arrow(x, y)
            for type_2 in set_basic_types:
                new_type2 = Arrow(type_, type_2)
                set_types.add(new_type2)

        # Replace Primitive with Polymorphic types with their instanciated counterpart
        for P in self.list_primitives[:]:
            type_P = P.type
            set_basic_types_P, set_polymorphic_types_P = type_P.decompose_type()
            if set_polymorphic_types_P:
                set_instantiated_types: Set[Type] = set()
                set_instantiated_types.add(type_P)
                for poly_type in set_polymorphic_types_P:
                    new_set_instantiated_types: Set[Type] = set()
                    for type_ in set_types:
                        for instantiated_type in set_instantiated_types:
                            unifier = {str(poly_type): type_}
                            intermediate_type = copy.deepcopy(instantiated_type)
                            new_type = intermediate_type.unify(unifier)
                            if new_type.size() <= upper_bound_type_size:
                                new_set_instantiated_types.add(new_type)
                    set_instantiated_types = new_set_instantiated_types
                for type_ in set_instantiated_types:
                    instantiated_P = Primitive(P.primitive, type=type_)
                    self.list_primitives.append(instantiated_P)
                self.list_primitives.remove(P)

    def __eq__(self, o: object) -> bool:
        return isinstance(o, DSL) and set(self.list_primitives) == set(
            o.list_primitives
        )

    def parse_program(self, program: str, type_request: Type) -> Program:
        """
        Parse a program from its string representation given the type request.
        """
        if " " in program:
            parts = list(
                map(lambda p: self.parse_program(p, type_request), program.split(" "))
            )

            def parse_stack(l: TList[Program]) -> Program:
                if len(l) == 1:
                    return l[0]
                current = l.pop(0)
                if isinstance(current.type, Arrow):
                    args = [parse_stack(l) for _ in current.type.arguments()]
                    return Function(current, args)
                return current

            sol = parse_stack(parts)
            assert str(sol) == program
            return sol
        else:
            program = program.strip("()")
            for P in self.list_primitives:
                if P.primitive == program:
                    return P
            if program.startswith("var"):
                varno = int(program[3:])
                vart = type_request
                if isinstance(type_request, Arrow):
                    vart = type_request.arguments()[varno]
                return Variable(varno, vart)
            assert False, f"can't parse: {program}"
