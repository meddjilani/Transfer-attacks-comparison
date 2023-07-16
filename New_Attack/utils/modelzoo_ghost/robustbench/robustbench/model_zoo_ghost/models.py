from collections import OrderedDict
from typing import Any, Dict, Dict as OrderedDictType


from utils.modelzoo_ghost.robustbench.robustbench.model_zoo_ghost.enums import BenchmarkDataset, ThreatModel
from utils.modelzoo_ghost.robustbench.robustbench.model_zoo_ghost.cifar10  import cifar_10_models


ModelsDict = OrderedDictType[str, Dict[str, Any]]
ThreatModelsDict = OrderedDictType[ThreatModel, ModelsDict]
BenchmarkDict = OrderedDictType[BenchmarkDataset, ThreatModelsDict]

model_dicts: BenchmarkDict = OrderedDict([
    (BenchmarkDataset.cifar_10, cifar_10_models),
])
