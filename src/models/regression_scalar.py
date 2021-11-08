import copy

import pytorch_lightning as pl
import torch
from torchvision.transforms import Compose

from src.data import transforms as tf
from src.models.util import _conc_outputs
from src.models.metrics import regression_metrics
from src.util.MAPS import CRITERIONS_MAP, OPTIMIZER_MAP, SCHEDULER_MAP
from src.util.build import build_model_from_config, build_dataloader_from_config, build_regression_normalizer, \
    build_criterion_from_config, get_gpu_augmentations


class RegressionModuleWithScalars(pl.LightningModule):
    """
    PyTorch Lightning Module for regression with additional scalar input
    """
    def __init__(self,
                 model_conf: dict,
                 loader_conf: dict,
                 crit: str,
                 optim: str,
                 optim_kwargs: dict,
                 crit_kwargs: dict = None,
                 scheduler: str = None,
                 scheduler_kwargs: dict = None,
                 scheduler_metric: str = None,
                 gpu_augmentations: dict = None,
                 target_normalizer: str = 'linear',
                 scalar_normalizer: str = 'linear'
                 ):
        """

        :param model_conf: Model configuration
        :param loader_conf: DataLoader configuration
        :param crit: Loss function
        :param optim: Optimizer
        :param optim_kwargs: Optimizer arguments
        :param crit_kwargs: Loss function arguments
        :param scheduler: Learning rate scheduler
        :param scheduler_kwargs: Learning rate scheduler arguments
        :param scheduler_metric: ReduceLROnPLateau metric
        :param gpu_augmentations: List of augmentations applied on GPU
        :param target_normalizer: Normalizer for regression targets
        :param scalar_normalizer: Normalizer for additional scalar input
        """
        super(RegressionModuleWithScalars, self).__init__()
        self.model = build_model_from_config(model_conf)
        self.train_loader, self.val_loader = build_dataloader_from_config(loader_conf)
        self.regression_normalizer = build_regression_normalizer(target_normalizer, loader_conf)
        tmp_conf = copy.deepcopy(loader_conf)
        tmp_conf['target_conf'] = "psa_pre"
        self.scalar_normalizer = build_regression_normalizer(scalar_normalizer, tmp_conf)
        self.crit = build_criterion_from_config(
            CRITERIONS_MAP[crit], crit_kwargs, self.train_loader
        ) if crit_kwargs else CRITERIONS_MAP[crit]()
        self.train_augs = get_gpu_augmentations(
            gpu_augmentations) if gpu_augmentations else lambda x: x
        self.val_augs = Compose(
            [tf.TensorToCupy(), tf.PerSampleNormalize(), tf.CupyToTensor()]) if gpu_augmentations else lambda x: x

        self.optim = optim

        self.scheduler = scheduler
        self.scheduler_kwargs = scheduler_kwargs
        self.scheduler_metric = "val/" + scheduler_metric
        self.save_hyperparameters(optim_kwargs)

    def forward(self, img) -> torch.Tensor:
        return self.model(img)

    def training_step(self, batch, batch_idx):
        X, y = batch
        image = X['image']
        for i in range(len(image)):
            X[i] = self.train_augs(image[i])
        X['image'] = image.unsqueeze(1) if len(image.shape) == 4 else image
        X['psa'] = self.scalar_normalizer.normalize(X['psa']).unsqueeze(1)
        y = self.regression_normalizer.normalize(y)
        out = self.model(X).squeeze(dim=1)
        loss = self.crit(out, y)
        return {'loss': loss, 'preds': out.detach().cpu(), 'targets': y.detach().cpu()}

    def on_train_start(self) -> None:
        self.logger.log_hyperparams(self.hparams, {"hp/mae": 0, "hp/rmse": 0})

    def training_epoch_end(self, outputs, key='train/'):
        y_pred, y_true, loss = _conc_outputs(outputs)
        metrics = regression_metrics(y_true, y_pred, self.regression_normalizer)
        metrics['loss'] = loss.mean()
        mapped = {}
        for k in metrics.keys():
            mapped[key + k] = metrics[k]
        if key == "val/":
            mapped['hp/mae'] = metrics['mae']
            mapped['hp/rmse'] = metrics['rmse']
        del metrics
        self.log_dict(mapped, logger=True, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        X, y = batch
        image = X['image']
        for i in range(len(image)):
            X[i] = self.val_augs(image[i])
        X['image'] = image.unsqueeze(1) if len(image.shape) == 4 else image
        X['psa'] = self.scalar_normalizer.normalize(X['psa']).unsqueeze(1)
        y = self.regression_normalizer.normalize(y)
        out = self.model(X).squeeze(dim=1)
        loss = self.crit(out, y)
        return {'loss': loss, 'preds': out.detach().cpu(), 'targets': y.detach().cpu()}

    def validation_epoch_end(self, outputs):
        self.training_epoch_end(outputs, key="val/")

    def configure_optimizers(self):
        optim = OPTIMIZER_MAP[self.optim]
        optim = optim(self.parameters(), **self.hparams)
        if not self.scheduler:
            return optim

        scheduler = SCHEDULER_MAP[self.scheduler](optim, **self.scheduler_kwargs)
        return {
            'optimizer': optim,
            'lr_scheduler': scheduler,
            'monitor': self.scheduler_metric,
            'strict': True,
        }

    def predict_step(self, batch, batch_idx, dataloader_idx = None):
        X, y = batch
        image = X['image']
        for i in range(len(image)):
            X[i] = self.val_augs(image[i])
        X['image'] = image.unsqueeze(1)
        X['psa'] = self.scalar_normalizer.normalize(X['psa']).unsqueeze(1)
        out = self.model(X).squeeze(dim=1)
        y = self.regression_normalizer.normalize(y)
        return {'pred': out, 'target': y, 'pre_psa':X['psa']}



    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader