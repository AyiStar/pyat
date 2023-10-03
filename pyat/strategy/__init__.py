from .base import BaseStrategy
from .random import RandomStrategy
from .us import MaximumEntropyStrategy
from .mfi import MaximumFisherInformationStrategy
from .kli import KullbackLeiblerInformationStrategy
from .maat import MAATStrategy
from .mamge import MAMGEStrategy
from .bald import BayesianActiveLearningByDisagreementStrategy

from .build import build_strategy
