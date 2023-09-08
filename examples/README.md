# Examples

This folder contains ready to use scripts and files that you can leverage to reproduce results from papers for example or to test your new ideas.

<!-- toc -->

- [Generics](#generics)
- [Programming By Example](#programming-by-example)
- [Programming from Natural Language](#programming-from-natural-language)
- [SyGuS](#sygus)

## Generics

Some scripts wield the same name across different kind of specifications, they often have the same interface and provide the same features.
Here is a list with explanations of such scripts:

- ``dataset_explorer.py`` this scripts takes a dataset as an input and enables yo uto explore different statistics about the dataset or to view specific tasks.

## Programming By Example

This the PBE folder. The specification of task is given as pairs of inputs-outputs of correct executions of the solution program.
The available domains are:
  
- integer list manipulation with deepcoder and dreamcoder;
- regexp;
- trandsuctions.

## Programming from Natural Language

This is the NLP folder. The specification of the task is given as a natural language string that explains the task.

## SyGuS

This is the SyGuS folder. It provides scripts to directly work with the SyGus format. The goal is to mainly edit the specification thanks to the tools offered by ProgSynth such as sharpening.
