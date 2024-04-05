import neptune.new as neptune
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.loggers.base import rank_zero_experiment


class NeptuneWrapperLogger(LightningLoggerBase):

    def __init__(self, run: neptune.Run):
        super().__init__()
        self._run = run

    @property
    def name(self):
        return 'NeptuneWrapperLogger'

    @property
    @rank_zero_experiment
    def experiment(self):
        return self._run

    @property
    def version(self):
        return '0.0'

    @rank_zero_only
    def log_hyperparams(self, params):
        self._run['params'] = params

    @rank_zero_only
    def log_metrics(self, metrics, step):
        for name, value in metrics.items():
            self._run[f'logs/{name}'].log(value, step=step)

    @rank_zero_only
    def finalize(self, status):
        self._run.stop()
        super().finalize(status)