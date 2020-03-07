import json

from ray.tune.schedulers import FIFOScheduler, AsyncHyperBandScheduler, \
    HyperBandScheduler, PopulationBasedTraining, TrialScheduler
from ray.tune.suggest import BasicVariantGenerator, SearchAlgorithm
from ray.tune.suggest.ax import AxSearch
from ray.tune.suggest.bayesopt import BayesOptSearch
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.suggest.nevergrad import NevergradSearch
from ray.tune.suggest.sigopt import SigOptSearch


# TODO: need to validate parameters, for conditions such as batch size with buffer memory
class AgentConfig:
    def __init__(self, config):
        self.config = config

    def __str__(self):
        return json.dumps(self.config, indent=4, default=str)


class TuneConfig:
    def __init__(self, config):
        self.config = config.copy()
        self.config['scheduler'] = self._validate_scheduler()
        self.config['search_alg'] = self._validate_search_alg()

    def _validate_scheduler(self) -> TrialScheduler:
        config = self.config['scheduler']
        schedulers = {
            'FIFO': FIFOScheduler,
            'AsyncHyperBand': AsyncHyperBandScheduler,
            'HyperBand': HyperBandScheduler,
            'PBT': PopulationBasedTraining,
        }
        scheduler = schedulers[config.get('type', 'FIFO')]
        return scheduler(**config.get('kwargs', dict()))

    def _validate_search_alg(self) -> SearchAlgorithm:
        config = self.config['search_alg']
        search_algorithms = {
            'AxSearch': AxSearch,
            'BasicVariantGenerator': BasicVariantGenerator,
            'BayesOptSearch': BayesOptSearch,
            'HyperOptSearch': HyperOptSearch,
            'NevergradSearch': NevergradSearch,
            'SigOptSearch': SigOptSearch,
        }
        search = search_algorithms[config.get('type', 'BasicVariantGenerator')]
        return search(**config.get('kwargs', dict()))

    def __str__(self):
        return json.dumps(self.config, indent=4, default=str)
