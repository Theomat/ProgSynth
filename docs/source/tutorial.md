# Tutorial

In this section, an overview of how to create a DSL is shown, based on the example of the *calculator* DSL, whose source code can be found in the folder ``./examples/pbe/calculator``.

## DSL Structure

Several files have been implemented in order to create a DSL allowing int-to-int or float-to-float additions and substractions.

* ``calculator/calculator.py``, contains the DSL implementation;
* `calculator/convert_calculator` is a runnable python script which enables you to convert the original calculator dataset file to the ProgSynth format;
* `calculator/calculator_task_generator`, an adapted version of the file `task_generator` that allows us to create a synthetic dataset, based on the one created with `convert_calculator`.

A single already existing file has to be modified in `./examples/pbe` and this tutorial will explain how to change it later.

### DSL (`calculator/calculator.py`)

The DSL is mainly represented by two dictionaries, representing the semantic and the syntax of the DSL.
Both associate to a string- the name of the primitive- either  its semantic or its type.

#### Syntax

A more detailed explanation of the type system is available in [Type System](type_system.md).

Primitives in ProgSynth are strongly typed, therefore you need to specify their types when declaring a new primitive.
Note that ProgSynth does not check that the actual data you are passing has any consistency with the declared type.

If you know what OCaml types look like, you can use ``auto_type("my_ocaml_type_syntax")``to automatically create your desired type, there are some differences such as tuples that we do not support.

The addition and substraction primitives should accept either 2 ints or 2 floats as parameters, we wish to define a custom type that can represent both.

A function is logically represented by arrows, indicating the parameters of the primitive and the output.
Either you can use arrows: `Arrow(type, Arrow(type, type))`
 or the `FunctionType` helper `FunctionType(type, type, type)`, both are equivalent to `type -> type -> type` or you can use the magic helper ``auto_type("type -> type -> type")``.

**Example**:
    `int2float` takes an int as input and will return a float. Thus, its type is `int -> float`.

ProgSynth uses custom-defined types:

* `PrimitiveType`, this is a base type solely defined by its name. ProgSynth already defines as `PrimitiveType` the following: INT, BOOL, STRING, UNIT. In our case, INT was already defined by ProgSynth. We simply need to define the `PrimitiveType` FLOAT for our DSL.
* `PolymorphicType`, these types can be any type that can be encoutered in the DSL. In practice ProgSynth will generate all different versions of the polymorphic type, remove the useless ones and add them in the DSL. To see how to constrain a polymorphic type see the detailed explanation of the [type system](type_system.md). It is also uniquely identified by its name.
  
  * As we have defined 2 `PrimitiveType`, the methods `+` and `-` will each be developped in two different versions: one allowing operations between integers and another one allowing operations between floats, there is no int and float version because it is the same polymorphic type object used.
  * As the DSL needs to know which `PrimitiveType`s can be encountered in order to replace any `PolymorphicType`, we define some constants in our DSL with the correct type (at least one per type).

#### Semantics

Simply put, the semantics are short snippets of code that will be executed when the primitive is used. This enables ProgSynth to combine these **primitives** in order to synthetise the program that we want to obtain.

For instance, as we want to solve tasks made of additions and substractions between two numbers of the same type, we only need to define only 3 primitives: +, - and int2float and the constants.

Here, we consider that an integer is a sub-type of float. Thus, we only need in this case to convert integers to float numbers using int2float.

To sum up, you can define the semantics for a primitive by creating a key associated with its name where the value is the actual function to execute or the value if it is a constant.
Observe that + is a binary function but the value associated is `lambda a: lambda b: round(a+b, 1)`, in order for ProgSynth to partially apply functions in Python, you must use only provide unary functions. As of now, we have not yet found a way to automatically convert functions to unary functions without a significant performance hit during evaluation so you will have to do it yourself.

#### Lexicon

In order to generate tasks and in order to use neural networks, a lexicon is needed for the DSL, a lexicon is a list of all base values that can be encountered in the DSL.
 Here, we limit our DSL to float numbers rounded to one decimal, in the range [-256.0, 257[.
For example, if our DSL were to manipulate lists of int or float, we would not have to add anything to the lexicon since lists are not a base type (`PrimitiveType`).

#### Forbidden patterns (Optional)

In some cases, we wish to stop the DSL to derive a specific primitive from another one:
For instance, let us say that we want to extend the `calculator` DSL with a primitive to `add1` to an integer and another one to `sub1` to an integer.
Because doing `(add1 (sub1 x))` or `(sub1 (add1 x))` is the same as doing nothing, we can forbid the second pattern from being derived from the first one, in that case the patterns would be `{ "add1": {"sub1}, "sub1": {"add1"}`.
For more information see [sharpening](sharpening.md) which describes in much more details the different capabilities available to reduce the size of the grammar, more powerful mechanisms are available.

## Creating a dataset (`calculator/convert_calculator.py`)

We need to create a dataset. For this example, we created a short JSON file named `dataset/calculator_dataset.json` that is built with the following fields:

* *program*, that contains therepresentation of the program, the parsing is done automatically from the DSL object (`dsl.parse`) so you don't need to parse it yourself. Here is a representation of a program that computes `f(x, y)= x + y * x` in our DSL: `(+ var0 (* var0 var1))`;

* *examples*, displaying what are the expected inputs and outputs of the program.

Once the dataset is done, we need to create a file converting it to the ProgSynth format, done here in `convert_calculator.py`.
An important point to note is that we need to develop the `PolymorphicType`, as described in the previous sub-section.

It is done automatically by calling the method `dsl.instantiate_polymorphic_types(upper_bound)`.
As we only want to develop `+` and `-` as methods with a size of 5 (INT -> INT -> INT or FLOAT -> FLOAT -> FLOAT, 3 types + 2 arrows = 5), we define its upper bound type size to 5.


If you want to adapt the code of `calculator/convert_calculator` for your own custom DSL, it should work almost out of the box with ProgSynth, note that ProgSynth needs to guess your type request and it does so from your examples. If you are manipulating types that are not guessed by ProgSynth, it wil fill them with UnknownType silently, in that case you may need to add your own function to guess type request or modify the one from ProgSynth which is `synth/syntax/type_system.py@guess_type`.

### Usage

We can simply use this file by command line, from the folder `./examples/pbe/calculator`.

```bash
python convert_calculator.py dataset/calculator_dataset.json -o calculator.pickle
```

## Generating a synthetic dataset

Once the DSL and a short dataset are created, we wish to generate automatically a dataset reproducing the task distribution.

The dataset generator works out of the box for our DSL but that may not always be the case, you can check out other DSLs files and look at the `task_generator_*.py` files.

### Usage

Once the DSL has been added to the `examples/pbe/dsl_loader.py` then you can generate datasets using:

```bash
python dataset_generator_unique.py --dsl calculator --dataset calculator/calculator.pickle -o dataset.pickle --inputs 1 --programs 1000
```

The generated dataset can be explored using `dataset_explorer.py`.

```bash
python dataset_explorer.py --dsl calculator --dataset dataset.pickle
```

## Conclusion

Once the dataset and the DSL are done, we simply need to add our DSL to the `dsl_loader.py` script, in-depth instructions are provided in the file. Then, the usage is the same as described in the section [usage](usage.rst).
