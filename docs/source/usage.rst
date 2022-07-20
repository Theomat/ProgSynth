Usage of ProgSynth
==================

The ProgSynth framework currently focuses on the Programming By Exemples (PBE) specification of tasks. Other cases are planned to be supported, despite this most elements of ProgSynth can be used independently of the specification.

The :code::`synth` folder is a standalone library and does not provide ready to use code such as :code:`model.fit()`, instead we provide in :code:`./examples` different elements:

- scripts that can directly be used;
- implementations of some DSLs.

These scripts enable you to reproduce the results from papers or can be modified to test yoru ideas. The script are pretty generic and in general can be used for your own custom DSL witt=h little to no modification.

For further information, in each specifcation folder inside  :code:`./examples` there is a :code:`README` explaining the use of scripts, what DSLs are implemented from which paper and where to download datasets.

The tutorial on section :doc:`tutorial` uses the example of tasks based on additions and substractions between integers or between floating point numbers, explaining step-by-step how to create a new DSL that can be used by the framework.

