# Tutorial

This tutorial will show you how to:

- [create a new DSL from scratch](#create-a-dsl-from-scratch);
- [add a semantic to the DSL](#add-a-semantic-to-the-dsl);

afterward everything is PBE specific:

- [add a DSL to the already existing PBE pipeline](#making-your-dsl-usable-by-scripts);
- [create your first dataset for this DSL](#creating-a-dataset);
- [explore a dataset](#explore-a-dataset);
- [create a task generator for that pipeline](#creating-a-task-generator);
- [generate synthetics datasets](#generating-a-synthetic-dataset);
- [train a model](#train-a-model);
- [evaluate a model](#evaluate-a-model);
- [synthesize a program](#simple-synthesis).

This example is the *calculator* DSL, whose source code can be found in the folder ``./examples/pbe/calculator``.

## Create a DSL from scratch

A DSL is a syntactic object thus it only defines the syntax of our primitives.
The relevant file is ``calculator/calculator.py``.

A primitive is a function or a constant that you might need to use in the solution program, it is typed and usually has a semantic, but the semantic is not defined in the DSL.

The syntax is a mapping from primitives to their type.

For detailed information about types [see the page on the type system](type_system.md).

The syntax object is a dictionnary where keys are unique strings identifying your primitives and values are ProgSynth types. It might be a bit long to explain all the different type features supported by ProgSynth, however ProgSynth provides the ``auto_type`` function which dramatically speed up the syntax writing process.
Here is an example:

```python
from synth.syntax import auto_type, DSL

syntax = auto_type({
  "1": "int",
  "2": "int",
  "3": "int",
  "3.0": "float"
  "int2float": "int -> float",
  "+": "'a [int | float] -> 'a [int | float] -> 'a [int | float]",
  "-": "'a [int | float] -> 'a [int | float] -> 'a [int | float]",
})

dsl = DSL(syntax)
```

The notation might seem complex to you but we will briefly explain what happens.
When you put a string such as ``"int"`` or ``"float`` then this is transformed into a ground type, the ``"->"`` translates into a function.
The ``"'"`` prefix tells us ``"'a"`` is a polymorphic type; however the ``[int | float]`` right after that indicates this polymorphic type can only take the following values: ``int`` or ``float``.
So that means we will have a ``+`` only for ``int`` and a ``+`` only for ``float``, but both will share the same semantic, since they both are named ``"+"``.
For detailed information about types and on how this works [see the page on the type system](type_system.md).

You can now use your DSL to generate [grammars](grammars.md)!

You might want to add *syntactic constraints* on the generated grammars, this is covered in [sharpening](sharpening.md).

## Add a semantic to the DSL

The relevant file is ``calculator/calculator.py``.
It's great that we can produce grammars and everything with our DSL but we cannot execute our program! It is time to gave them a semantic!
The semantic object is a dictionnary where keys are unique strings identifying your primitives and values are unary functions or constants.

Here is the semantic for the primitives we defined earlier:

```python
from synth.semantic import DSLEvaluator

semantic = {
    "+": lambda a: lambda b: round(a + b, 1),
    "-": lambda a: lambda b: round(a - b, 1),
    "int2float": lambda a: float(a),
    "1": 1,
    "2": 2,
    "3": 3,
    "3.0": 3.0,
}

evaluator = DSLEvaluator(semantic)
```

First for constants, they are just associated to their value.
Then for functions, notice that while ``+`` is a binary function, here we have a unary function that returns another unary function.
ProgSynth needs functions in unary form in order to be able to do partial applications.
Python's system to automatically transform a n-ary function to a unary function as of now induces a relatively high execution cost, which makes it prohibitive for ProgSynth.

You can now use your evaluator to eval your program, the syntax is ``evaluator.eval(program, inputs_as_a_list)``.

As a side note, it might happen that in your evaluation, exceptions occur and you do not want to interrupt the python process, in that case you can use ``evaluator.skip_exceptions.add(My_Exception)``. When such an exception occurs, it is caught and instead a ``None`` is returned.

The evaluator cached the evaluation of programs, so the value is computed only once on the same input. However, in some cases, you might need to clear the cache since it can take a lot of space which can be done using: ``evaluator.clear_cache()``.

---
**Everything after is PBE specific.**

---

## Making your DSL usable by scripts

Most if not all scripts in the ``pbe`` folder should work with little to no changes for most DSLs.
These scripts use the ``dsl_loader.py`` file that manages DSLs and provides a streamline approach for all scripts to load and use them.
You should add your DSL to that script to be able to use all these scripts for free.

But since this is PBE specific we need to define a lexicon in ``calculator/calculator.py``.

### Lexicon

In the PBE specification, a lexicon is needed in order to:

- create synthetic tasks and thus synthetic datasets;
- use neural networks for prediction.

A lexicon is a list of all base values that can be encountered in the DSL.
 Here, we limit our DSL to float numbers rounded to one decimal, in the range [-256.0, 257[.
For example, if our DSL were to manipulate lists of int or float, we would not have to add anything to the lexicon since lists are not a base type (`PrimitiveType`).

### Finally adding your DSL

Your only point of interest in this file is the ``__dsl_funcs`` dictionnary that should be surrounded by comments.
Here is the line that we added to the dictionnary for our calculator DSL:

```python
"calculator": __base_loader(
        "calculator.calculator",
        [
            "dsl",
            "evaluator",
            "lexicon",
            ("reproduce_calculator_dataset", "reproduce_dataset"),
        ],
    ),
```

It tells the loader that the DSL is defined in the file ``calculator/calculator.py``, then when it loads this file, it loads the following variables ``dsl, evaluator, lexicon, reproduce_calculator_dataset``.
These variables will be made available under the following fields respectively ``dsl, evaluator, lexicon, reproduce_dataset``.
Notice that the tuple notation allows renaming.
The first three are necessary while the last one is optional, in the sense that you might not need to redefine a ``reproduce_dataset`` function.

## Creating a dataset

The relevant file is ``calculator/convert_calculator.py``.

To generate a synthethic dataset we need to create a dataset.
For this example, we created a short JSON file named `dataset/calculator_dataset.json` that is built with the following fields:

- *program*: that contains the representation of the program, the parsing is done automatically by the DSL object (`dsl.parse`) so you don't need to parse it yourself. Here is a representation of a program that computes `f(x, y)= x + y * x` in our DSL: `(+ var0 (* var1 var0))`;
- *examples*: displaying what are the expected inputs and outputs of the program.

Once the dataset is done, we need to create a file converting it to the ProgSynth format, done here in `convert_calculator.py`.
An important point to note is that we need to develop the `PolymorphicType`, since our ``+`` and ``-`` depend on it so before parsing we need to call `dsl.instantiate_polymorphic_types()`.

If you want to adapt the code of `calculator/convert_calculator` for your own custom DSL, it should work almost out of the box with ProgSynth, note that ProgSynth needs to guess your type request and it does so from your examples. If you are manipulating types that are not guessed by ProgSynth, it wil fill them with ``UnknownType`` silently, in that case you may need to add your own function to guess type request or modify the one from ProgSynth which is `synth/syntax/type_helper.py@guess_type`.

We can simply use this file by command line, from the folder `./examples/pbe/calculator`.

```bash
python convert_calculator.py dataset/calculator_dataset.json -o calculator.pickle
```

## Explore a dataset

You might want to check that you correctly translated your task to the ProgSynth format.
This can be done easily by visualizing the tasks of a dataset with the dataset explorer.
A dataset can be explored using `dataset_explorer.py`.

```bash
python examples/pbe/dataset_explorer.py --dsl calculator --dataset calculator.pickle
```

## Creating a Task Generator

Most often you don't need to use a custom TaskGenerator and the default one will work, however if you have more than one ground type you will need to do so, this is the case with the calculator DSL.
The code is at the end of ``calculator/calculator.py``.

**TODO: explain in more details**

## Generating a synthetic dataset

Now we can create synthetic datasets.
There is already existing script that does all of the job for us.

The dataset generator works out of the box for our DSL but that may not always be the case, you can check out other DSLs files and look at the `task_generator_*.py` files.

You can generate datasets using:

```bash
python examples/pbe/dataset_generator_unique.py --dsl calculator --dataset calculator/calculator.pickle -o dataset.pickle --inputs 1 --programs 1000
```

## Train a model

For more information about model creation see [this page](prediction.md).

You can easily train a model using:

```bash
python examples/pbe/model_trainer.py --dsl calculator --dataset my_train_dataset.pickle --seed 42 --b 32 -o my_model.pt -e 2
```

There are various options to configure your model and everything which we do not dwelve into.

## Infer with a model

A model can be used to produce PCFGs, this will produce a `pickle` file in the same folder as your model, you will need to pass this file to the solver.

```bash
python examples/pbe/model_prediction.py --dsl calculator --dataset my_test_dataset.pickle --model my_model.pt --b 32 -support my_train_dataset.pickle
```

The ``--support my_train_dataset.pickle`` is only used to filter the test set on type requests that were also present in the train set.

## Evaluate a model

You might want to evaluate a model to see if it learned anything relevant, this can be easily done but is time consuming.
To evaluate a model, we actually try to solve program synthesis tasks for a DSL so this is not simply an inference task.
If you are directly interested in synthesizing your first program then jump over to [the next section](#simple-synthesis) which tells you exactly how to do that.
You can easily evaluate a model using:

```bash
python examples/pbe/solve.py --dsl calculator --dataset my_test_dataset.pickle --pcfg pcfgs_my_test_dataset_my_model.pt -o . -t 60 --support my_train_dataset.pickle --solver cutoff
```

The most important parameter is perhaps ``-t 60`` which gives a timeout of 60 seconds per task.
You can also play with different solver, by default ``cutoff`` works pretty well on almost anything.

This will produce a CSV file in the output folder (``.`` above).
This result file can then be plotted using:

```bash
python examples/plot_solve_results.py --dataset my_test_dataset.pickle --folder . --support my_train_dataset.pickle
```

Again there's a plethora of options available, so feel free to play with them.

## Simple synthesis

Here is a simple function that takes your task, the PCFG and the evaluator and generates a synthetised program.
For more information about predictions and how to produce a P(U)CFG from a model, see [this page](prediction.md).
If you are perhaps more interested in solving then you should probably look at the files in ``synth.pbe.solvers`` which offer different ways of solving our synthesis problem.

```python
from synth import Task, PBE
from synth.semantic import DSLEvaluator
from synth.syntax import bps_enumerate_prob_grammar, ProbDetGrammar
from synth.pbe import CutoffPBESolver

def synthesis(
    evaluator: DSLEvaluator,
    task: Task[PBE],
    pcfg: ProbDetGrammar,
    task_timeout: float = 60
):
    solver = CutoffPBESolver(evaluator)
    solution_generator = solver.solve(task, bps_enumerate_prob_grammar(pcfg), task_timeout)
    try:
        solution = next(solution_generator)
        print("Solution:", solution)
    except StopIteration:
        # Failed generating a solution
        print("No solution found under timeout")
    for stats in solver.available_stats():
        print(f"\t{stats}: {solver.get_stats(stats)}")
    
```
