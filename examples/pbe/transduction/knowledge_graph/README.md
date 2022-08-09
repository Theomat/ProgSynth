
# How to reproduce the experiment?

First you need a database that supports SPARQL queries.
Once you have that, you can generate the database using the ``fill_knowledge_graph.sparql``.

Then you need to execute ``convert_kg_json_tasks.py`` to convert ``constants.json`` to ``constants.pickle`` (supported by AutoSynth).
Then you need to preprocess the tasks with ``preprocess_tasks.py`` which will guess the constants.
Then you can use the ``constants.pickle`` file in ``evaluate.py`` with your model.

In our paper the model was the one obtained through the scrip ``test_performance.sh`` in experiments.
