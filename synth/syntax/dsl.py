import copy
from typing import Callable, Dict, Mapping, Optional, List as TList, Set, Tuple

from synth.syntax.type_system import UNIT, Type, Arrow, List
from synth.syntax.program import Function, Primitive, Program, Variable


class DSL:
    """
    Object that represents a domain specific language

    Parameters:
    -----------
    - syntax: maps primitive names to their types
    - forbidden_patterns: forbidden local derivations

    """

    def __init__(
        self,
        syntax: Mapping[str, Type],
        forbidden_patterns: Optional[Dict[Tuple[str, int], Set[str]]] = None,
    ):
        self.list_primitives = [
            Primitive(primitive=p, type=t) for p, t in syntax.items()
        ]
        self.forbidden_patterns = forbidden_patterns or {}
        self._forbidden_computed = False

    def __str__(self) -> str:
        s = "Print a DSL\n"
        for P in self.list_primitives:
            s = s + "{}: {}\n".format(P, P.type)
        return s

    def instantiate_polymorphic_types(self, upper_bound_type_size: int = 10) -> None:
        """
        Must be called before compilation into a grammar or parsing.
        Instantiate all polymorphic types.

        Parameters:
        -----------
        - upper_bound_type_size: maximum type size of type instantiated for polymorphic types
        """
        # Generate all basic types
        set_basic_types: Set[Type] = set()
        for P in self.list_primitives:
            set_basic_types_P, set_polymorphic_types_P = P.type.decompose_type()
            set_basic_types = set_basic_types | set_basic_types_P
        if UNIT in set_basic_types:
            set_basic_types.remove(UNIT)

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
            _, set_polymorphic_types_P = type_P.decompose_type()
            if set_polymorphic_types_P:
                set_instantiated_types: Set[Type] = set()
                set_instantiated_types.add(type_P)
                for poly_type in set_polymorphic_types_P:
                    new_set_instantiated_types: Set[Type] = set()
                    for type_ in set_types:
                        if not poly_type.can_be(type_):
                            continue
                        for instantiated_type in set_instantiated_types:
                            unifier = {str(poly_type): type_}
                            intermediate_type = copy.deepcopy(instantiated_type)
                            new_type = intermediate_type.unify(unifier)
                            if new_type.size() <= upper_bound_type_size:
                                new_set_instantiated_types.add(new_type)
                    set_instantiated_types = new_set_instantiated_types
                for type_ in set_instantiated_types:
                    instantiated_P = Primitive(P.primitive, type=type_)
                    if instantiated_P not in self.list_primitives:
                        self.list_primitives.append(instantiated_P)
                self.list_primitives.remove(P)

        # Duplicate things for Sum types
        for P in self.list_primitives[:]:
            versions = P.type.all_versions()
            if len(versions) > 1:
                for type_ in versions:
                    instantiated_P = Primitive(P.primitive, type=type_)
                    self.list_primitives.append(instantiated_P)
                self.list_primitives.remove(P)

        # Now remove all UNIT as parameters from signatures
        for P in self.list_primitives[:]:
            if any(arg == UNIT for arg in P.type.arguments()):
                P.type = P.type.without_unit_arguments()

    def __eq__(self, o: object) -> bool:
        return isinstance(o, DSL) and set(self.list_primitives) == set(
            o.list_primitives
        )

    def parse_program(self, program: str, type_request: Type) -> Program:
        """
        Parse a program from its string representation given the type request.

        Parameters:
        -----------
        - program: the string representation of the program, i.e. str(prog)
        - type_request: the type of the requested program in order to identify variable types

        Returns:
        -----------
        A parsed program that matches the given string
        """
        if " " in program:
            parts = list(
                map(lambda p: self.parse_program(p, type_request), program.split(" "))
            )
            function_calls: TList[int] = []
            level = 0
            levels: TList[int] = []
            elements = program.split(" ")
            for element in elements:
                if level > 0:
                    function_calls[levels[-1]] += 1
                function_calls.append(0)
                if element.startswith("("):
                    level += 1
                    levels.append(len(function_calls) - 1)
                end = 1
                while element[-end] == ")":
                    level -= 1
                    end += 1
                    levels.pop()

            def parse_stack(l: TList[Program], function_calls: TList[int]) -> Program:
                if len(l) == 1:
                    return l[0]
                current = l.pop(0)
                f_call = function_calls.pop(0)
                if current.type.is_instance(Arrow) and f_call > 0:
                    args = [
                        parse_stack(l, function_calls)
                        for _ in current.type.arguments()[:f_call]
                    ]
                    return Function(current, args)
                return current

            sol = parse_stack(parts, function_calls)
            assert (
                str(sol) == program
            ), f"Failed parsing:{program} got:{sol} type request:{type_request} obtained:{sol.type}"
            return sol
        else:
            program = program.strip("()")
            for P in self.list_primitives:
                if P.primitive == program:
                    return P
            if program.startswith("var"):
                varno = int(program[3:])
                vart = type_request
                if type_request.is_instance(Arrow):
                    vart = type_request.arguments()[varno]
                return Variable(varno, vart)
            assert False, f"can't parse: {program}"

    def get_primitive(self, name: str) -> Optional[Primitive]:
        """
        Returns the Primitive object with the specified name if it exists and None otherwise

        Parameters:
        -----------
        - name: the name of the primitive to get
        """
        for P in self.list_primitives:
            if P.primitive == name:
                return P
        return None

    def instantiate_semantics(
        self, semantics: Dict[str, Callable]
    ) -> Dict[Primitive, Callable]:
        """
        Transform the semantics dictionnary from strings to primitives.
        """
        dico = {}
        for key, f in semantics.items():
            for p in self.list_primitives:
                if p.primitive == key:
                    dico[p] = f
        return dico

    def __or__(self, other: "DSL") -> "DSL":
        out = DSL({})
        out.list_primitives += self.list_primitives
        for prim in other.list_primitives:
            if prim not in self.list_primitives:
                out.list_primitives.append(prim)
        out.forbidden_patterns = {k: v for k, v in self.forbidden_patterns.items()}
        for k, v in other.forbidden_patterns.items():
            if k not in out.forbidden_patterns:
                out.forbidden_patterns[k] = v

        return out
