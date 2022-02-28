from synth.specification import PBE
from synth.task import Dataset

d: Dataset[PBE] = Dataset.load("./dataset.pickle")
for i in range(len(d)):
    print(d[i])
    print()
