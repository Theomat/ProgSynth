Installation
==========================

From Source
------------

If you are installing from source, you will need Python 3.7.1 or later.

Install ProgSynth
-----------------

We recommend you to use a virtual environment whenever you install a git project. This can be done using the following command:

.. code:: bash

    python -m venv py3

This will create a folder called `py3` that will contain your python virtual environment. You can then activate it using the `source` command.

.. code:: bash

    source py/bin/activate


You can later leave this virtual environment using `deactivate`.

ProgSynth depedencies
---------------------

ProgSynth can be installed from source with `pip`, `conda` or `poetry`. 

.. code:: bash

    pip install .


or, if poetry is installed in your environment (recommended),

.. code:: bash

    poetry install
