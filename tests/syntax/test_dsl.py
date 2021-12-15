from synth.syntax.dsl import DSL
from synth.syntax.type_system import INT, FunctionType, PolymorphicType


syntax = {
    "+": FunctionType(INT, INT, INT),
    "id": FunctionType(PolymorphicType("a"), PolymorphicType("a")),
}


def test_instantiate_polymorphic() -> None:
    polymorphic_keys = [key for key, t in syntax.items() if t.is_polymorphic()]
    for size in [2, 5, 7, 11]:
        dsl = DSL(syntax)
        dsl.instantiate_polymorphic_types(size)

        for primitive in dsl.list_primitives:
            type = primitive.type
            assert not type.is_polymorphic()
            print(primitive.primitive)
            if primitive.primitive in polymorphic_keys:
                assert type.size() <= size

        cpy = DSL(syntax)
        cpy.instantiate_polymorphic_types(size)
        cpy.instantiate_polymorphic_types(size)
        assert cpy == dsl
