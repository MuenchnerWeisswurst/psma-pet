import json
import os
import sys

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin

from src.models.classification import ClassificationModule
from src.models.regression import RegressionModule
from src.models.regression_scalar import RegressionModuleWithScalars
from src.util.MAPS import *


torch.set_num_threads(16)


def tune_and_fit(model, sconfig):
    logger = TensorBoardLogger(os.path.join("runs", sconfig['experiment_name']), default_hp_metric=False)
    lrmonitor = LearningRateMonitor(logging_interval="epoch", log_momentum=True)
    checkpoint = ModelCheckpoint(save_last=True)
    n_gpus = sconfig.get('n_gpus', 1)
    trainer = pl.Trainer(gpus=n_gpus,
                         precision=16,
                         accelerator="ddp" if n_gpus > 1 else None,
                         plugins=DDPPlugin(find_unused_parameters=False) if n_gpus > 1 else None,
                         auto_lr_find=sconfig.get('auto_lr', False),
                         max_epochs=sconfig['epochs'],
                         logger=logger,
                         check_val_every_n_epoch=1,
                         callbacks=[lrmonitor, checkpoint],
                         stochastic_weight_avg=sconfig.get('stochastic_weight_avg', False),
                         )
    trainer.tune(model)
    trainer.fit(model)
    with open(os.path.join("runs", sconfig['experiment_name'], 'config.json'), 'w') as fd:
        json.dump(sconfig, fd, indent=4)
    del model
    del trainer


def cnn_classification(sconfig: dict):
    model = ClassificationModule(
        model_conf=sconfig['model'],
        loader_conf=sconfig['train_loader'],
        crit=sconfig['crit'],
        crit_kwargs=sconfig.get('crit_kwargs', None),
        optim=sconfig['optim'],
        optim_kwargs=sconfig['optim_kwargs'],
        scheduler=sconfig.get('scheduler', None),
        scheduler_metric=sconfig.get('scheduler_metric', None),
        scheduler_kwargs=sconfig.get('scheduler_kwargs', None),
        gpu_augmentations=sconfig.get('gpu_augmentations', None)
    )
    tune_and_fit(model, sconfig)


def cnn_regression(sconfig: dict):
    model = RegressionModule(
        model_conf=sconfig['model'],
        loader_conf=sconfig['train_loader'],
        crit=sconfig['crit'],
        crit_kwargs=sconfig.get('crit_kwargs', None),
        optim=sconfig['optim'],
        optim_kwargs=sconfig['optim_kwargs'],
        scheduler=sconfig.get('scheduler', None),
        scheduler_metric=sconfig.get('scheduler_metric', None),
        scheduler_kwargs=sconfig.get('scheduler_kwargs', None),
        target_normalizer=sconfig.get('regression_normalizer', 'linear'),
        gpu_augmentations=sconfig.get('gpu_augmentations', None)
    )

    tune_and_fit(model, sconfig)


def cnn_regression_with_scalars(sconfig: dict):
    model = RegressionModuleWithScalars(
        model_conf=sconfig['model'],
        loader_conf=sconfig['train_loader'],
        crit=sconfig['crit'],
        crit_kwargs=sconfig.get('crit_kwargs', None),
        optim=sconfig['optim'],
        optim_kwargs=sconfig['optim_kwargs'],
        scheduler=sconfig.get('scheduler', None),
        scheduler_metric=sconfig.get('scheduler_metric', None),
        scheduler_kwargs=sconfig.get('scheduler_kwargs', None),
        target_normalizer=sconfig.get('regression_normalizer', 'linear'),
        gpu_augmentations=sconfig.get('gpu_augmentations', None)
    )
    tune_and_fit(model, sconfig)


def main(config_path=None):
    with open(config_path, 'r') as fd:
        sconfig = json.load(fd)

    if 'type' not in sconfig.keys():
        cnn_classification(sconfig)
    elif sconfig['type'] == 'cnn_classification':
        cnn_classification(sconfig)
    elif sconfig['type'] == 'cnn_regression':
        cnn_regression(sconfig)
    elif sconfig['type'] == 'cnn_regression_with_scalars':
        cnn_regression_with_scalars(sconfig)
    else:
        raise NotImplementedError(f"Unsupported training type {sconfig['type']}")
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main(sys.argv[1])
    exit(0)
