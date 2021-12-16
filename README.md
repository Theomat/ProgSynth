![ProgSynth Logo](./images/logo.png)

--------------------------------------------------------------------------------

ProgSynth is a high-level framework that enbales to leverage program synthesis for other domains such as reinforcement learning or system design.

<!-- toc -->

- [More About ProgSynth](#more-about-progsynth)
  - [Combining Deep Learning with Theoretical Guarantees](#combining-deep-learning-with-theoretical-guarantees)
  - [A Scalable Framework](#a-scalable-framework)
- [Installation](#installation)
  - [From Source](#from-source)
    - [Install ProgSynth](#install-progsynth)
- [Getting Started](#getting-started)
- [Tools](./tools)
- [The Team](#the-team)
- [License](#license)

<!-- tocstop -->

## More About ProgSynth

At a granular level, ProgSynth is a library that consists of the following components:

| Component | Description |
| ---- | --- |
| [**synth**](./synth) | The high level synthesis libary |
| [**synth.generation**](./synth/generation) | A compilation of tools to generate objetcs needed for the synthesis, it is mainly used with deep learning  |
| [**synth.pbe**](./synth/pbe) | A library to work in the Programming By Example (PBE) framework |
| [**synth.semantic**](./synth/semantic) | The library of program evaluators |
| [**synth.syntax**](./synth/syntax) | The library to manipulate dsl, grammars, probabilistic grammars |

Elaborating Further:

### Combining Deep Learning with Theoretical Guarantees

The advantage of "classic" algorithms are their theoretical guarantees.
But many new deep learning based methods have emerged, they provide a tremendous efficiency but lose almost all theoretical guarantees.
ProgSynth provides already implemented algorithms that combine both approaches to get he best of both worlds: speed and guarantees!

### A Scalable Framework

Computing is now done at a large scale in a parallelilized fashion.
As such frameworks should also adapt: they should scale with more computing power but also leverage the power of parallelization.
This was taken into account and this is why for most algorithms we provide, we also provide a way to scale with the number of available processors.

For example, the `ConcretePCFG` can be split into disjoint sub `ConcretePCFG` to split the enumeration of the grammar into multiple jobs thus enabling to scale linearly with the numbers of workers.

## Installation

### From Source

If you are installing from source, you will need Python 3.7.1 or later.

#### Install ProgSynth

ProgSynth can be installed from source with `pip`, `conda`, `poetry`.

```bash
pip install .
```

## Getting Started

You can observe the [tools](./tools) made available, they should provide you with examples use of ProgSynth.

## The Team

ProgSynth is a project initiated by [Nathanaël Fijalkow](https://nathanael-fijalkow.github.io/) and joined by [Théo Matricon](https://theomat.github.io/).

## License

ProgSynth has a MIT license, as found in the [LICENSE](LICENSE) file.
