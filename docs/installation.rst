Installation
************

Installing Python 3.9
---------------------
The latest version of isopy requires python 3.9 and above.

Below is a guide to install python 3.9 if you are using the
[Anaconda](https://www.anaconda.com/products/individual) or
[miniconda](https://docs.conda.io/en/latest/miniconda.html)
distributions.

.. rubric:: New environment

Python 3.9 is not yet supported by all python libraries. I would
therefore recommend creating a new python environment for python 3.9.
To do this launch the anaconda prompt (Avaliable from the Anaconda
launcher).

If you havent done so already now is a good time to add the
[conda-forge](https://conda-forge.org/) repository channel. This channel
*may* be more up to date than the default channel.

.. code-block:: bash

    conda config --add channels conda-forge
    conda config --set channel_priority strict


Use the following command to create a new python environment replacing
``<myenv>`` with the name of your new environment:

.. code-block:: bash

    conda create -n <myenv> python 3.9


To active your new environment and install packages from the conda
repository type:

.. code-block:: bash

    conda activate <myenv>
    conda install <package-name>


.. rubric:: Upgrading existing environment

If you wish to upgrade your existing environment use the following
command with your preferred environment activated:

.. code-block:: bash

    conda install python=3.9



.. note::

    Not all packages support python 3.9 yet and the installation might
    fail. Additionally if you have a lot of packages installed, such as the
    full anaconda distribution, it might take a **long** time to complete.


Installing isopy
----------------
isopy is can be installed using `pip <https://pip.pypa.io/en/stable/>`_.
If you are using anaconda you will have to open the Anaconda Prompt
and activate the environment you wish to install isopy into. Then you
can install isopy using the following command:

.. code-block:: bash

    pip install isopy

Upgrading isopy
---------------

To upgrade to the newest version of isopy use the ``--upgrade`` flag:

.. code-block:: bash

    pip install --upgrade isopy

