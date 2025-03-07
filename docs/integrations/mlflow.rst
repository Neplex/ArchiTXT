MLFlow
======

MLFlow is an open-source platform for managing the end-to-end machine learning lifecycle, including experiment tracking, model versioning, and deployment.
With ArchiTXT’s integration, you can effortlessly log experiment executions, traces, and key metrics directly to MLFlow for streamlined monitoring and analysis.

ArchiTXT creates a dedicated MLFlow experiment for logging and every simplification process is recorded as a separate run in this experiment.
This detailed logging allows you to:

- Track the progress and performance of each simplification step.
- Compare different runs to identify the most effective parameters.
- Easily navigate through experiment histories using the MLFlow UI.

Configure MLFlow
^^^^^^^^^^^^^^^^

.. tip::

    By default, MLFlow logs experiments to a local directory.
    It is the recommended solution if you just want to try MLFlow.

To connect ArchiTXT to a remote MLFlow tracking server, set the environment variable `MLFLOW_TRACKING_URI`:

.. code-block:: sh

   export MLFLOW_TRACKING_URI=http://127.0.0.1:5000  # Replace with your remote host

Alternatively, you can set the tracking URI directly in your Python code:

.. code-block:: python

   import mlflow

   mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Replace with your remote host

Run Experiments
^^^^^^^^^^^^^^^

ArchiTXT automatically logs experiments to MLFlow as you run them.
Simply execute your experiments as usual, and ArchiTXT will handle the logging of runs and metrics.

Explore Data
^^^^^^^^^^^^

Visualize your logged data using the MLFlow web interface.
To start the MLFlow UI locally, run:

.. code-block:: sh

    mlflow ui

Open your browser and navigate to the default URL (usually `<http://127.0.0.1:5000>`_) to review your experiment details and performance metrics.

For more details on MLFlow and its capabilities, please refer to the `MLFlow Documentation <https://www.mlflow.org/docs/latest/index.html>`_.
