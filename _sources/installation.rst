Installation
============

There are multiple ways to install ArchiTXT:

- :ref:`Using a Python package manager (Recommended) <installation:Stable Release (Recommended)>` – Use pip, Poetry, PDM, uv, or another tool.
- :ref:`Installing the development version <installation:Development Version>` – Get the latest updates from GitHub.
- :ref:`Using a container <installation:Container Installation>` – Run ArchiTXT in a Docker or Podman container.

Python Installation
-------------------

.. margin::

    .. epigraph::

       "There's a snake in my boot!"

       -- Woody, *Toy Story*

ArchiTXT is available as a python package on **PyPI** (`pypi.org/project/architxt <https://pypi.org/project/architxt>`_).
You can install it using any Python package manager of your choice.
It requires **Python 3.10+**, to check your Python version, run:

.. tab-set::

    .. tab-item:: Unix/macOS

        .. code-block:: sh

            $ python3 --version

    .. tab-item:: Windows

        .. code-block:: sh

            $ py --version

If you don't know about Python packaging, you can read the `official guide <https://packaging.python.org/en/latest/tutorials/installing-packages/>`_.


Stable Release (Recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To install the latest stable release of ArchiTXT, use one of the following methods:

.. tab-set::

    .. tab-item:: pip

        .. code-block:: sh

            $ pip install "architxt"

        To install a specific version

        .. code-block:: sh

            $ pip install "architxt==<version>"

    .. tab-item:: Poetry

        .. code-block:: sh

            $ poetry add "architxt"

        To install a specific version

        .. code-block:: sh

            $ poetry add "architxt==<version>"

    .. tab-item:: PDM

        .. code-block:: sh

            $ pdm add "architxt"

        To install a specific version

        .. code-block:: sh

            $ pdm add "architxt==<version>"

    .. tab-item:: uv

        .. code-block:: sh

            $ uv add "architxt"

        To install a specific version

        .. code-block:: sh

            $ uv add "architxt==<version>"


Development Version
^^^^^^^^^^^^^^^^^^^

To install the latest development version directly from GitHub:

.. code-block:: sh

    $ pip install git+https://github.com/Neplex/ArchiTXT.git

This version may include new features and bug fixes that have not yet been released.


Container Installation
----------------------

ArchiTXT is also available as a pre-built container image.

.. tab-set::

    .. tab-item:: Docker

        Pull the latest **Docker** image:

        .. code-block:: sh

            $ docker pull ghcr.io/neplex/architxt:latest

        Start the UI:

        .. code-block:: sh

            $ docker run -d \
                -e CORENLP_URL=http://corenlp-uri:9000 \
                -p 8080:8080 \
                --name architxt \
                ghcr.io/neplex/architxt:latest

        Use the CLI directly:

        .. code-block:: sh

            $ docker run --rm \
                -e CORENLP_URL=http://corenlp-uri:9000 \
                --name architxt \
                ghcr.io/neplex/architxt:latest \
                --help

    .. tab-item:: Podman

        Pull the latest **Podman** image:

        .. code-block:: sh

            $ podman pull ghcr.io/neplex/architxt:latest

        Start the UI:

        .. code-block:: sh

            $ podman run -d \
                -e CORENLP_URL=http://corenlp-uri:9000 \
                -p 8080:8080 \
                --name architxt \
                ghcr.io/neplex/architxt:latest

        Use the CLI directly:

        .. code-block:: sh

            $ podman run --rm \
                -e CORENLP_URL=http://corenlp-uri:9000 \
                --name architxt \
                ghcr.io/neplex/architxt:latest \
                --help

    .. tab-item:: Compose

        .. code-block:: yaml

            services:
                architxt:
                    image: ghcr.io/neplex/architxt:latest
                    ports:
                        - "8080:8080"
                    environment:
                        CORENLP_URL: http://corenlp-uri:9000

---------------------

Once installed, you're ready to :doc:`Get Started! <getting_started>`
