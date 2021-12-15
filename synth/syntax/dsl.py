import copy
from typing import Mapping, Optional, Set

from synth.syntax.type_system import Type, Arrow, List
from synth.syntax.program import Primitive


class DSL:
    """
    Object that represents a domain specific language

    list_primitives: a list of primitives.

    """

    def __init__(
        self,
        syntax: Mapping[str, Type],
        no_repetitions: Optional[Set[str]] = None,
    ):
        self.list_primitives = [
            Primitive(primitive=p, type=t) for p, t in syntax.items()
        ]
        self.no_repetitions: Set[str] = no_repetitions or set()

    def __repr__(self) -> str:
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
