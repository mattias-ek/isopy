Installation
************

Installing Python 3.9
---------------------
The latest version of isopy requires python 3.9 and above.

Below is a guide to install python 3.9 if you are using the
`Anaconda <https://www.anaconda.com/products/individual>`_ or
`miniconda <https://docs.conda.io/en/latest/miniconda.html>`_
distributions.

.. rubric:: New conda environment

Python 3.9 is not yet supported by all python libraries. I would
therefore recommend creating a new python environment for python 3.9.
To do this launch the anaconda prompt (Avaliable from the Anaconda
launcher).

If you havent done so already now is a good time to add the
`conda-forge <https://conda-forge.org/>`_ repository channel. This channel
*may* be more up to date than the default channel.

.. code-block:: bash

    conda config --add channels conda-forge
    conda config --set channel_priority strict

If you are using `Jupyter <https://jupyter.org/>`_ also make sure that
you have the package ``nb_conda_kernels`` installed in your main conda
environment. If not install in your default environment using:

.. code-block::

    conda install nb_conda_kernels

This will allow you to use your other python environments in Jupyter.

Use the following command to create a new python environment replacing
``<myenv>`` with the name of your new environment:

.. code-block:: bash

    conda create -n <myenv> python=3.9

This will create a bare bones python environment. You will need to
manually install any libraries you want to use. If you are using Jupyter
install ``ipykernel`` to use this environment for your notebooks.
Make sure to activate your new environment before installing new
libraries:

.. _activate:

.. code-block:: bash

    conda activate <myenv>
    conda install <package-name>

Isopy is not yet avaliable on conda so you will need to install it
using pip as described in `Installing isopy`_.

.. rubric:: Upgrading conda environment

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
and activate_ the environment you wish to install isopy into. Then you
can install isopy using the following command:

.. code-block:: bash

    pip install isopy

This will also install libraries that isopy depend on such as
`numpy <https://numpy.org/>`_, `scipy <https://www.scipy.org/>`_
and `matplotlib <https://matplotlib.org/>`_.

Upgrading isopy
---------------

To upgrade to the newest version of isopy use the ``--upgrade`` flag:

.. code-block:: bash

    pip install --upgrade isopy

