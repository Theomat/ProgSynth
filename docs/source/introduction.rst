Introduction
============

Combining Deep Learning with Theoretical Guarantees
----------------------------------------------------

The advantage of "classic" algorithms are their theoretical guarantees.
But many new deep learning based methods have emerged, they provide a tremendous efficiency but lose almost all theoretical guarantees.
ProgSynth provides already implemented algorithms that combine both approaches to get he best of both worlds: speed and guarantees!

A Scalable Framework
--------------------

Computing is now done at a large scale in a parallelilized fashion.
As such frameworks should also adapt: they should scale with more computing power but also leverage the power of parallelization.
This was taken into account and this is why for most algorithms we provide, we also provide a way to scale with the number of available processors.

For example, the `ConcretePCFG` can be split into disjoint sub `ConcretePCFG` to split the enumeration of the grammar into multiple jobs thus enabling to scale linearly with the numbers of workers.

The Team 
--------
ProgSynth is a project initiated by [Nathanaël Fijalkow](https://nathanael-fijalkow.github.io/) and joined by [Théo Matricon](https://theomat.github.io/).

License
--------
ProgSynth has a MIT license, as found in the :doc:`license` file.
