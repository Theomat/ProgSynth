Tutorial
========

In this section, an overview of how to create a DSL is shown, based on the example of the *calculator* DSL, which source code can be found in the folder :code:`./examples/pbe/calculator`.

Structure of a DSL
------------------

Several files have been implemented in order to create a DSL allowing int-to-int or float-to-float additions and substractions.

* :code:`calculator/calculator.py`, containing the primitives of the DSL and the default evaluator.
* :code:`calculator/convert_calculator` is a runnable python script which enables you to convert the original calculator dataset file to the ProgSynth format.
* :code:`calculator/calculator_task_generator`, an adapted version of the file :code:`task_generator` that allows us to create a synthetic dataset, based from the one created with :code:`convert_calculator`.

Some changes have to be made to other files present in :code:`./examples/pbe`, and this tutorial will display how we can change them to adapt a new DSL.

DSL (:code:`calculator/calculator.py`)
--------------------------------------
The DSL is mainly represented by two dictionaries, representing the semantics of the DSL and the types of its methods.

Semantics of the DSL
~~~~~~~~~~~~~~~~~~~~
Simply put, the semantics are the short snippets of code that will compose your program. This will allow the DSL to combine these **primitives** in order to synthetise the program that we want to obtain.

For instance, as we want to solve tasks that necessit to make additions and substarctions between to real numbers of the same type, we mainly need to define only 3 methods: +, - and int2float.

Here, we consider that an integer is a sub-type of float. Thus, we only need in this case to convert integers to float numbers using int2float.

.. _Types of the DSL:

Types of the DSL
~~~~~~~~~~~~~~~~
A DSL have to be strongly typed in order to properly work. As we wish the addition and substraction methods to accept either 2 ints or 2 floats as parameters, we wish to define a custom type that can represent both.

A method is logically represented by arrows, indicating the parameters of the primitive and the output.

**Example**::
    :code:`int2float` takes an int as input and will return a float. Thus, its type is :code:`int -> float`

ProgSynth framework uses custom-defined types, that can be

* :code:`PrimitiveType`, where the type is solely defined by its name.
  
  - In our case, INT was already defined in the framework. We simply need to define the :code:`PrimitiveType` FLOAT for our DSL.
* :code:`PolymorphicType`, where the DSL will replace later-on this type with every :code:`PrimitiveType` that can be encountered. 
  
  - As we have defined 2 :code:`PrimitiveType`, the methods :code:`+` and :code:`-` will each be developped in two different versions: one allowing operations between integers, and another one allowing operations between floats.
  - As the DSL needs to know which :code:`PrimitiveType`s can be encountered in order to replace any :code:`PolymorphicType`, we define some constants in our DSL with the correct type (at least one per type).

Lexicon
~~~~~~~
A DSL is defined inside a specific lexicon. To avoid overflow or underflow, we limit our DSL to float numbers rounded to one decimal, in the range [-256.0, 257[

Forbidden patterns (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In some cases, we wish to stop the DSL to derive a specific primitive from another one::
    For instance, let us say that we want to extend the `calculator` DSL with a primitive to `add1` to an integer and another one to `sub1` to an integer.
    Because doing `(add1 (sub1 0))` or `(sub1 (add1 0))`, we can create a list of pairs of patterns where the second pattern cannot be derived from the first one. 


Creating a dataset (:code:`calculator/convert_calculator.py`)
--------------------------------------------------------------
To use PBE, we need to create a dataset. For this example, we created a short .json file named :code:`dataset/calculator_dataset.json` that is constructed with the following fields

* *program*, that contains the stack representing the program with its inputs and its primitives. Each part of the program is separated by a pipe :code:`|`
  
  - :code:`INT` or :code:`FLOAT` are our :code:`PrimitiveType` and represent the inputs of the program
  - The other words are the primitives of the program. If the primitives are methods and require inputs, we indicate where to find the inputs. For instance, let our program allowing us to remove 2 to a float be :code:`FLOAT|2|int2float,1|-,0,2`
  
    + :code:`int2float` will convert the element at position 1: our primitive :code:`2`
    + :code:`-` will substract to the element at position 0 (an input of type :code:`FLOAT`) the element at position 2 (the converted value of the primitive :code:`2`)

* *examples*, displaying what are the expected inputs and outputs of the program.

Once the dataset is done, we need to create a file converting it to the ProgSynth format, done here in :code:`convert_calculator.py`.
An important point to note is that we need to develop the :code:`PolymorphicType`, as described in the previous sub-section.

It is done automatically by calling the method :code:`dsl.instantiate_polymorphic_types()` or the dsl created.
As we only want to develop :code:`+` and :code:`-` as methods with a size of 5 (INT -> INT -> INT or FLOAT -> FLOAT -> FLOAT, as we consider each arrow in the size), we define its upper bound type size to 5.


Structures used
~~~~~~~~~~~~~~~

If you want to adapt the code of :code:`calculator/convert_calculator` for your own custom DSL, the part that will need to be changed is stored inside :code:`__calculator_str2prog()`, that returns the final program and its type (inputs and outputs).

To do so, useful structures defined by the file :code:`synth/syntax/program.py` are available. At the moment, please refrain from using classes :code:`Lambda` and :code:`Constant`, as they are not fully implemented in the framework yet.

In order to properly construct the returned type, it is important to add to :code:`type_stack` only the methods and the inputs of the program, as hinted in the section :ref:`Types of the DSL`.


Usage
~~~~~
We can simply use this file by command line, from the folder :code:`./examples/pbe/calculator`.

.. code:: bash

    python convert_calculator.py dataset/calculator_dataset.json -o calculator.pickle


Generating a synthetic dataset (:code:`dataset_generator.py`)
-------------------------------------------------------------
Once the DSL and a short dataset are created, we wish to generate automatically a dataset reproducing the task distribution. We have to adapt the file :code:`dataset_generator.py` for this.

The *deepcoder* and *dreamcoder* datasets did not require to use float numbers. Thus, the previous implementation of the :code:`task_generator.py` needs to be adapted to float numbers.
Hence, we create the file :code:`calculator/calculator_task_generator.py` and change some methods (and, if required, some imports).

* the method :code:`basic_output_validator` needs to allow outputs of type float
* the method :code:`reproduce_dataset` 
  - needs to analyse the range of both int and float inputs and to create a sampler for each type, with the corresponding specific lexicon. 
  - must return a :code:`TaskGenerator` object that has the correct program lexicon. Here, as the int lexicon is included in the float lexicon, we return the latter one.

Usage
~~~~~
Once this file is created and is properly imported in :code:`dataset_generator.py`, we can use it by command line, from the folder :code:`./examples/pbe`.
Do not forget that you have to adapt the arguments of this file to the DSL you created before using it.

.. code:: bash

    python dataset_generator.py --dsl calculator --dataset calculator/calculator.pickle -o dataset.pickle

The dataset generated can be explored using :code:`dataset_explorer.py`.

.. code:: bash
    
    python dataset_explorer.py --dsl calculator --dataset dataset.pickle


Conclusion
----------
Once the dataset and the DSL are done, we simple need to adapt the imports of :code:`pcfg_prediction.py`, :code:`evaluate.py` and :code:`dsl_analyser.py`. Then, the usage is the same as describe in the section :doc:`usage`.
