Tutorial
========

In this section, an overview of how to create a DSL is shown, based on the example of the *calculator* DSL, whose source code can be found in the folder :code:`./examples/pbe/calculator`.

Structure of a DSL
------------------

Several files have been implemented in order to create a DSL allowing int-to-int or float-to-float additions and substractions.

* :code:`calculator/calculator.py`, containing the primitives of the DSL and the default evaluator.
* :code:`calculator/convert_calculator` is a runnable python script which enables you to convert the original calculator dataset file to the ProgSynth format.
* :code:`calculator/calculator_task_generator`, an adapted version of the file :code:`task_generator` that allows us to create a synthetic dataset, based from the one created with :code:`convert_calculator`.

Some changes have to be made to other files present in :code:`./examples/pbe`, and this tutorial will display how to change them for a new DSL.

DSL (:code:`calculator/calculator.py`)
--------------------------------------
The DSL is mainly represented by two dictionaries, representing the semantics of the DSL and the types of its primitives.

Semantics of the DSL
~~~~~~~~~~~~~~~~~~~~
Simply put, the semantics are short snippets of code that will make up your programs. This enables ProgSynth to combine these **primitives** in order to synthetise the program that we want to obtain.

For instance, as we want to solve tasks made of additions and substractions between two numbers of the same type, we only need to define only 3 primitives: +, - and int2float.

Here, we consider that an integer is a sub-type of float. Thus, we only need in this case to convert integers to float numbers using int2float.

.. _Types of the DSL:

Types of the DSL
~~~~~~~~~~~~~~~~
A DSL has to be strongly typed in order to properly work. The addition and substraction primtives should accept either 2 ints or 2 floats as parameters, we wish to define a custom type that can represent both.

A method is logically represented by arrows, indicating the parameters of the primitive and the output.

**Example**::
    :code:`int2float` takes an int as input and will return a float. Thus, its type is :code:`int -> float`

ProgSynth uses custom-defined types:

* :code:`PrimitiveType`, where the type is solely defined by its name.
  
  - In our case, INT was already defined by ProgSynth. We simply need to define the :code:`PrimitiveType` FLOAT for our DSL.
* :code:`PolymorphicType`, where the DSL will replace later-on this type with every :code:`PrimitiveType` that can be encountered. 
  
  - As we have defined 2 :code:`PrimitiveType`, the methods :code:`+` and :code:`-` will each be developped in two different versions: one allowing operations between integers and another one allowing operations between floats.
  - As the DSL needs to know which :code:`PrimitiveType`s can be encountered in order to replace any :code:`PolymorphicType`, we define some constants in our DSL with the correct type (at least one per type).

Lexicon
~~~~~~~
A DSL is defined inside a specific lexicon. To avoid overflow or underflow, we limit our DSL to float numbers rounded to one decimal, in the range [-256.0, 257[

Forbidden patterns (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In some cases, we wish to stop the DSL to derive a specific primitive from another one:
For instance, let us say that we want to extend the `calculator` DSL with a primitive to `add1` to an integer and another one to `sub1` to an integer.
Because doing `(add1 (sub1 x))` or `(sub1 (add1 x))` is the same as doing nothing, we can forbid the second pattern from being derived from the first one, in that case the patterns would be :code:`{ "add1": {"sub1}, "sub1": {"add1"}`.
For more information see `pruning <pruning.html>`_.


Creating a dataset (:code:`calculator/convert_calculator.py`)
--------------------------------------------------------------
To use PBE, we need to create a dataset. For this example, we created a short JSON file named :code:`dataset/calculator_dataset.json` that is built with the following fields

* *program*, that contains therepresentation of the program, the parsing is done automatically from the DSL object so you don't need to parse it yourself. Here ia representation of a program that computes :code:`f(x, y)= x + y * x` in our DSL: :code:`(+ var0 (* var0 var1))`.

* *examples*, displaying what are the expected inputs and outputs of the program.

Once the dataset is done, we need to create a file converting it to the ProgSynth format, done here in :code:`convert_calculator.py`.
An important point to note is that we need to develop the :code:`PolymorphicType`, as described in the previous sub-section.

It is done automatically by calling the method :code:`dsl.instantiate_polymorphic_types(upper_bound)`.
As we only want to develop :code:`+` and :code:`-` as methods with a size of 5 (INT -> INT -> INT or FLOAT -> FLOAT -> FLOAT, as we consider each arrow in the size), we define its upper bound type size to 5.


If you want to adapt the code of :code:`calculator/convert_calculator` for your own custom DSL, it should work almost out of the box with ProgSynth, some point that may hinder you is that ProgSynth needs to guess your type request, it does so from your examples. If you are manipulating types that are not guess by ProgSynth, it wil fill them with UnknownType silently, in that case you may need to add your own function to guess type request or modify the one from ProgSynth.


Usage
~~~~~
We can simply use this file by command line, from the folder :code:`./examples/pbe/calculator`.

.. code:: bash

    python convert_calculator.py dataset/calculator_dataset.json -o calculator.pickle


Generating a synthetic dataset (:code:`dataset_generator.py`)
-------------------------------------------------------------
Once the DSL and a short dataset are created, we wish to generate automatically a dataset reproducing the task distribution.

The *deepcoder* and *dreamcoder* datasets did not require to use float numbers. Thus, the previous implementation of the :code:`task_generator.py` needs to be adapted to float numbers.
Hence, we need to create a function to enable reproduing our dataset. We recommend checking the documentation of ``reproduce_dataset`` which we will use.

* the function :code:`analyser` needs to analyse the range of both int and float inputs, it is basically called on base types.
* the function :code:`get_element_sampler` produces the sampler for our base types, here uniform on our int and float ranges.
* the function :code:`get_validator` produces a function that takes an ouput produced from a program and should tell whether this output is allowed or not.
* the function :code:`get_lexicon` produces the lexicon of the DSL, it will be used for deep learning. Here, as the int lexicon is included in the float lexicon, we return the latter one.

Usage
~~~~~
Once the DSL has been added to the :code:`dsl_loader.py` then you can generate datasets using:
.. code:: bash

    python dataset_generator.py --dsl calculator --dataset calculator/calculator.pickle -o dataset.pickle

The dataset generated can be explored using :code:`dataset_explorer.py`.

.. code:: bash
    
    python dataset_explorer.py --dsl calculator --dataset dataset.pickle


Conclusion
----------
Once the dataset and the DSL are done, we simple need to add our DSL to the :code:`dsl_loader.py` script, in-depth instructions are provided in the file. Then, the usage is the same as describe in the section :doc:`usage`.
