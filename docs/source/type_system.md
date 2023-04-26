# The Type System

The type system in ProgSynth has the vocation of adding constraints for compilation from a DSL into a grammar.

ProgSynth does not check at any time that the data you manipulate has the correct type, types are only used at compilation time.

<!-- toc -->
Table of contents:

- [Basic Type](#basic-type)
- [Advanced Types](#advanced-types)
  - [Sum Type](#sum-type)
  - [Arrow](#arrow)
  - [Generic](#generic)
- [Polymorphic Types](#polymorphic-types)
- [Methods of interest](#methods-of-interest)

<!-- tocstop -->

## Basic Type

Ground types are ``PrimitiveType``.
An ``int`` is represented as ``PrimitiveType("int")``.
Notice that it is uniquely identified by its name therefore two instances with the same name represent the same type.

ProgSynth already defines the following types: ``INT``, ``BOOL``, ``STRING``, ``UNIT``.

## Advanced Types

Advanced types are built from other types.
There are three:

- Sum Type;
- Arrow;
- Generic.

### Sum Type

Sum types are union in python.
They can be built two ways, first with ``Sum(t1, t2)`` or ``t1 | t2``.

### Arrow

Arrow represent functions.
They can be built with ``Arrow(t1, t2)``.
A function ``int->int->int`` would have type ``Arrow(int, Arrow(int, int))``.
An easier way to construct these arrows is to use ``FunctionType(int, int, int)`` in this case. While ``Arrow`` is a binary constructor, ``FunctionType`` allows any number of arguments and ensure that the ``Arrow``are built correctly especially if you are using higher order functions.

### Generic

Generic are parametric types. For example, the previous type ``Arrow``is a generic, the ``List``type is also one. You can make a list type out of any type using ``List(t)``.
To instanciate a Generic builder you can use the ``GenericFunctor``:

```python
List = GenericFunctor("list", min_args=1, max_args=1)
Arrow = GenericFunctor(
    "->",
    min_args=2,
    infix=True,
)

```

## Polymorphic Types

One can instantiate polymorphic types with ``PolymorphicType("a")`` as with ``PrimitiveType`` they are uniquely identified by their name.
At compilation, a polymorphic type will take as possible values any ground type that is present in the DSL and advanced types built on top of them recursively up to some type size.

You can limit the set of types a polymorphic type can take with ``FixedPolymorphicType("f", t1, t2, t3)`` which will only take types that are ``t1``, ``t2`` or ``t3``.

You can check if a polymorphic type ``poly_t`` can be assigned some type ``t`` with ``poly_t.can_be(t)``.

## Methods of interest

Creating a type can be done by instanciating the objects individually or you can use the ``auto_type`` method which takes either your string type or your syntax dictionnary and transform it into a real type. Here are a few examples:

```python
from synth.syntax import auto_type

t = auto_type("int") 
# PrimitiveType("int")
t = auto_type("int | float -> float") 
# Arrow(int | float, float)
t = auto_type("'a list  ('a -> 'b ) -> 'b list") 
# let
# a = PolymorphicType("a")
# b = PolymorphicType("b")
# in
# FunctionType(List(a), Arrow(a, b), List(b))

t = auto_type("'a[int | float] -> 'a[int | float]")
# let 
# a = FixedPolymorphicType("a", PrimitiveType("int"), PrimitiveType("float))
# or equivalently 
# a = FixedPolymorphicType("a", PrimitiveType("int") | PrimitiveType("float))
# in
# Arrow(a, a)

t = auto_type("int optional")
# Generic("optional", PrimitiveType("int"))
```

- ``t1.is_instance(t2)`` computes wether type ``t1`` is an instance of ``t2``, notice that ``t2`` can be a python type, some type in our system or a ``TypeFunctor`` such as a ``GenericFunctor`` like ``List`` or ``Arrow``.
Note that there might be questions regarding covariant and contravariant types, but since our types are completely removed from the notion of subtype, those two notions are not relevant to our type system;
- ``t.arguments()`` returns the list of arguments consumed by this type if this is an arrow, if this not an arrow then an empty list is returned;
- ``t.returns()`` returns the type returned by this arrow, if this is not an arrow return the type ``t`` itself.
- ``t.all_versions()`` returns the list of all types that this type can take, this is only relevant for sum types, basically each sum type will take all its possible values;
- ``t.unify({'a': INT})`` will return ``t`` where all polymorphic types named ``'a'`` take value ``INT``.
