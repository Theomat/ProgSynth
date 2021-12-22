import os

hashseed = os.getenv("PYTHONHASHSEED")
if not hashseed:
    os.environ["PYTHONHASHSEED"] = "0"


from synth.task import Task, Dataset
from synth.specification import TaskSpecification, PBE, Example
