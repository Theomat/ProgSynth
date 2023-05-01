# Grammars

Grammars are programs generator, all our grammars are finite.
Typically when you instantiate a grammar you always specify a maximum depth.

The main object of interest are probabilistic grammars on which most methods to enumerate and sample programs are provided.

<!-- toc -->
Table of contents:

- [Grammar Models](#grammar-models)
  - [det-CFG](#det-cfg)
  - [U-CFG](#u-cfg)
- [Probabilistic Grammars](#probabilistic-grammars)

<!-- tocstop -->

## Grammar Models

Currently the only grammar model supported are [Context-free grammars](https://en.wikipedia.org/wiki/Context-free_grammar) (CFG).
All our rules have the following form:

```
S -> f S1 S2 ... Sk
S -> g
```

where ``S``, ``S1``, ..., ``Sk`` are non terminal and ``f`` is a primitive of arity ``k`` and ``g`` is a primitive of arity 0, in other words a constant.

We have two different models: deterministic CFG and unambiguous CFG; while the latter is more expressive it is around 20% slower but used correctly the gains are huge.

The ways to generate a grammar are mainly through static methods such as ``MyGrammarModel.depth_constraint(dsl, type_request)``.
Grammars albeit already complex objects are not the final object of interests in ProgSynth.
The most relevant methods are:

- ``program in grammar`` which returns whether program belongs to the grammar or not;
- ``grammar.programs()`` which yields the number of programs contained in the grammar, do not convert it to float as this easily yield values over MAX_DOUBLE, hence we return an int to take advantage of the lack of limit for int in python;
- ``grammar.derive(...)`` which allows you to derive your program step by step;
- ``grammar.derive_all(...)`` which derives the whole given subtree for you and hands you the result;
- ``grammar.reduce_derivatons(...)`` which is like a fold over the derivation steps of the given program.

### det-CFG

A CFG which has the following property:
> For a given non-terminal ``S``, for any primitive ``f``, there is at most one derivation from ``S`` using primitive ``f``

In other words, it is deterministic to derive ``f`` from non-terminal ``S``.

In ProgSynth this is the default model, that is ``CFG``.
If you do not use [sharpening](sharpening.md) for example, then ProgSynth uses this model when producing a grammar.

### U-CFG

A CFG which has the following property:
> For a tree/program ``t``, there exists at most one derivation for tree/program ``t`` in the grammar

In other words, there is no ambiguity to derive a program from the grammar, but locally it may be ambiguous, that is you have to try all derivation rules for the primitive to find out later which is the one that allows deriving the program.

``UCFG`` in ProgSynth can express all regular tree languages and is generated when you use [sharpening](sharpening.md).

## Probabilistic Grammars

We offer tagged grammars, those are grammars where derivations are tagged with a generic type, replacing 'probabilistic' with 'tagged' in what is following will work as well.
The most relevant one is when derivations are tagged with float giving you probabilistic grammars.
> For a given non-terminal ``S``, the set of all derivations from ``S`` make up a probability distribution, *i.e.* sum up to 1.

There are two models: ``ProbGrammar`` and ``ProbUGrammar`` respectively working for ``CFG`` and ``UCFG``.
Basically adding a U for class and a u_ for methods to the classic method will yield the equivalent methods for the unambiguous model.

Probabilistic grammars offer a wide range of interesting methods to generate programs:

- ``pgrammar.sample()`` sample a random program from the grammar, you will need to first call ``pgrammar.init_sampling(seed)`` for sampling to work, sampling is optimised compared to naive sampling;
- ``enumerate_prob_(u_)_grammar(pgrammar)`` which gives you an enumerator that will enumerate programs in the grammar by decreasing order of probability;
- ``split(pgrammar, n)`` which gives you ``n`` disjoint probabilistic unambiguous grammars that make up a partition of the original given ``pgrammar``, the main intereset is to easily parallelise the enumeration.

Of course, since probabilistic grammars are grammars they also offer the same methods as classic grammars.

**But I want to enumerate programs by size?**

Well, you can just use ``Prob(U)Grammar.uniform(grammar)`` and enumerate that probabilistic grammar will give you an enumeration by program size.
