MLFlow
======

MLFlow is an open-source platform for managing the end-to-end machine learning lifecycle, including experiment tracking, model versioning, and deployment.
With ArchiTXT’s integration, you can effortlessly log experiment executions, traces, and key metrics directly to MLFlow for streamlined monitoring and analysis.

.. seealso::

    `MLFlow Documentation <https://www.mlflow.org/docs/latest/index.html>`_
        The official documentation of MLFlow.

ArchiTXT creates a dedicated MLFlow experiment for logging and every simplification process is recorded as a separate run in this experiment.
This detailed logging allows you to:

- Track the progress and performance of each simplification step.
- Compare different runs to identify the most effective parameters.
- Easily navigate through experiment histories using the MLFlow UI.

Configure MLFlow
----------------

.. tip::

    By default, MLFlow logs experiments to a local directory.
    It is the recommended solution if you just want to try MLFlow.

To connect ArchiTXT to a remote MLFlow tracking server, set the environment variable `MLFLOW_TRACKING_URI`:

.. code-block:: sh

   $ export MLFLOW_TRACKING_URI=http://127.0.0.1:5000  # Replace with your remote host

You can also set the tracking URI in your Python code:

.. code-block:: python

   import mlflow

   mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Replace with your remote host

Run Experiments
---------------

ArchiTXT can log experiments to MLFlow if it is executed within an active MLFlow run.
In Python, you can create a run as follows:

.. code-block:: python

   import mlflow

   with mlflow.start_run():
       ... # <- Your code here

Once the run is started, execute your experiments as usual, and ArchiTXT will automatically handle the logging.

You can also enable MLFlow logging whe using the :doc:`CLI </getting_started/cli>` by using the ``--log`` option.

Explore Data
------------

Visualize your logged data using the MLFlow web interface.
If running locally, you can start the MLFlow UI by running:

.. code-block:: sh

    $ mlflow ui

Open your browser and navigate to the default URL (usually `<http://127.0.0.1:5000>`_).
In the web UI, you can review your experiment details and performance metrics.

Simplification
^^^^^^^^^^^^^^

During the simplification, the following metrics are logged to MLFlow by the :py:meth:`architxt.metrics.Metrics.log_to_mlflow` method after each ``iteration``.
Additional debug artifacts may be logged when ``debug=True``.

Parameters
""""""""""

+-----------------------------------------+--------------+------------------------------------------------------------------------------------------+
| Parameter Name                          | Range / Type | Description                                                                              |
+=========================================+==============+==========================================================================================+
| ``nb_sentences``                        | ``int``      | Total number trees in the forest.                                                        |
+-----------------------------------------+--------------+------------------------------------------------------------------------------------------+
| ``tau``                                 | ``[0, 1]``   | The threshold for subtree similarity.                                                    |
+-----------------------------------------+--------------+------------------------------------------------------------------------------------------+
| ``epoch``                               | ``int``      | The maximum number of iteration.                                                         |
+-----------------------------------------+--------------+------------------------------------------------------------------------------------------+
| ``min_support``                         | ``int``      | The minimum support for structures to be considered frequent.                            |
+-----------------------------------------+--------------+------------------------------------------------------------------------------------------+
| ``metric``                              | ``str``      | The name of the metric used for the tree similarity.                                     |
+-----------------------------------------+--------------+------------------------------------------------------------------------------------------+
| ``edit_ops``                            | ``str``      | The list of operations that will be applied on the trees.                                |
+-----------------------------------------+--------------+------------------------------------------------------------------------------------------+

General Metrics
"""""""""""""""

+-----------------------------------------+--------------+------------------------------------------------------------------------------------------+
| Metric Name                             | Range / Type | Description                                                                              |
+=========================================+==============+==========================================================================================+
| ``nodes.count``                         | ``int``      | Total number of nodes in the forest.                                                     |
+-----------------------------------------+--------------+------------------------------------------------------------------------------------------+
| ``unlabeled.count``                     | ``int``      | Number of nodes that have no associated label.                                           |
+-----------------------------------------+--------------+------------------------------------------------------------------------------------------+
| ``redundancy``                          | ``[0, 1]``   | Median redundancy score of attribute groups exceeding a functional dependency threshold. |
+-----------------------------------------+--------------+------------------------------------------------------------------------------------------+

Clustering
""""""""""

+-----------------------------------------+--------------+------------------------------------------------------------------------------------------+
| Metric Name                             | Range / Type | Description                                                                              |
+=========================================+==============+==========================================================================================+
| ``clustering.cluster_count``            | ``int``      | Number of distinct clusters in the current forest.                                       |
+-----------------------------------------+--------------+------------------------------------------------------------------------------------------+
| ``clustering.ami``                      | ``[-1, 1]``  | Adjusted Mutual Information between original and current clustering.                     |
+-----------------------------------------+--------------+------------------------------------------------------------------------------------------+
| ``clustering.completeness``             | ``[0, 1]``   | Measures if all members of a class are assigned to the same cluster.                     |
+-----------------------------------------+--------------+------------------------------------------------------------------------------------------+

Entities
""""""""

+-----------------------------------------+--------------+------------------------------------------------------------------------------------------+
| Metric Name                             | Range / Type | Description                                                                              |
+=========================================+==============+==========================================================================================+
| ``entities.coverage``                   | ``[0, 1]``   | Jaccard similarity between original and current entity sets.                             |
+-----------------------------------------+--------------+------------------------------------------------------------------------------------------+
| ``entities.count``                      | ``int``      | Total number of entity-type nodes.                                                       |
+-----------------------------------------+--------------+------------------------------------------------------------------------------------------+
| ``entities.distinct_count``             | ``int``      | Number of distinct entity labels.                                                        |
+-----------------------------------------+--------------+------------------------------------------------------------------------------------------+
| ``entities.ratio``                      | ``[0, 1]``   | Average number of entity nodes per distinct entity label.                                |
+-----------------------------------------+--------------+------------------------------------------------------------------------------------------+

Groups
""""""

+-----------------------------------------+--------------+------------------------------------------------------------------------------------------+
| Metric Name                             | Range / Type | Description                                                                              |
+=========================================+==============+==========================================================================================+
| ``groups.count``                        | ``int``      | Total number of group-type nodes.                                                        |
+-----------------------------------------+--------------+------------------------------------------------------------------------------------------+
| ``groups.distinct_count``               | ``int``      | Number of distinct group labels.                                                         |
+-----------------------------------------+--------------+------------------------------------------------------------------------------------------+
| ``groups.ratio``                        | ``[0, 1]``   | Average number of group nodes per distinct group label.                                  |
+-----------------------------------------+--------------+------------------------------------------------------------------------------------------+

Relations
"""""""""

+-----------------------------------------+--------------+------------------------------------------------------------------------------------------+
| Metric Name                             | Range / Type | Description                                                                              |
+=========================================+==============+==========================================================================================+
| ``relations.count``                     | ``int``      | Total number of relation-type nodes.                                                     |
+-----------------------------------------+--------------+------------------------------------------------------------------------------------------+
| ``relations.distinct_count``            | ``int``      | Number of distinct relation labels.                                                      |
+-----------------------------------------+--------------+------------------------------------------------------------------------------------------+
| ``relations.ratio``                     | ``[0, 1]``   | Average number of relation nodes per distinct relation label.                            |
+-----------------------------------------+--------------+------------------------------------------------------------------------------------------+

Collections
"""""""""""

+-----------------------------------------+--------------+------------------------------------------------------------------------------------------+
| Metric Name                             | Range / Type | Description                                                                              |
+=========================================+==============+==========================================================================================+
| ``collections.count``                   | ``int``      | Total number of collection-type nodes.                                                   |
+-----------------------------------------+--------------+------------------------------------------------------------------------------------------+
| ``collections.distinct_count``          | ``int``      | Number of distinct collection labels.                                                    |
+-----------------------------------------+--------------+------------------------------------------------------------------------------------------+
| ``collections.ratio``                   | ``[0, 1]``   | Average number of collection nodes per distinct collection label.                        |
+-----------------------------------------+--------------+------------------------------------------------------------------------------------------+

Schema
""""""

+-----------------------------------------+--------------+------------------------------------------------------------------------------------------+
| Metric Name                             | Range / Type | Description                                                                              |
+=========================================+==============+==========================================================================================+
| ``schema.overlap``                      | ``[0, 1]``   | Overlap ratio of attribute groups in the current schema.                                 |
+-----------------------------------------+--------------+------------------------------------------------------------------------------------------+
| ``schema.balance``                      | ``[0, 1]``   | Balance score of group sizes in the current schema.                                      |
+-----------------------------------------+--------------+------------------------------------------------------------------------------------------+
| ``schema.productions``                  | ``int``      | Number of productions (grammar rules) in the current schema.                             |
+-----------------------------------------+--------------+------------------------------------------------------------------------------------------+
| ``schema.non_terminal``                 | ``int``      | Number of non-terminal symbols (labels) in the current schema.                           |
+-----------------------------------------+--------------+------------------------------------------------------------------------------------------+
