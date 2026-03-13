import sys

from eval.navi.models import NaviContentInput, NaviContentOutput

sys.path.insert(0, "../opensbt-llm/")

from collections import defaultdict
from typing import Tuple, List, Dict, Optional

import numpy as np

from examples.navi.fitness import NaviFitnessContentComparison
from llm.config import N_VALIDATORS
from judge_eval.validator_dim import llm_validator_conversation
from opensbt.evaluation.fitness import Fitness
from llm.model.mt_simout import MultiTurnSimulationOutput
from llm.eval.fitness import counter_validations

from examples.navi.fitness_mt import NaviFitnessConversationEffectiveness, NaviFitnessConversationEfficiency, NaviFitnessConversationValidationDimensions
from llm.eval.critical import CriticalByFitnessThreshold, CriticalMerged
from llm.eval.fitness import FitnessMerged

fitness_fnc = FitnessMerged([
    NaviFitnessConversationValidationDimensions(),
    NaviFitnessConversationEfficiency(),
    NaviFitnessConversationEffectiveness(),
])

critical_fnc = CriticalMerged(
    fitness_names=fitness_fnc.name,
    criticals=[
        (CriticalByFitnessThreshold(mode = "<", score=0.7), ["dimensions_fitness"]),
        (CriticalByFitnessThreshold(mode = "<", score=0.7), ["efficiency_fitness"]),
        (CriticalByFitnessThreshold(mode = "<", score=0.7), ["effectiveness_fitness"]),
    ],
    mode="or",
)
