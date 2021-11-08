import pytorch_lightning as pl
import torch
from torchvision.transforms import Compose

from src.data import transforms as tf
from src.data.dataset import Slice2DDataset
from src.models.metrics import classification_metrics
from src.models.util import _conc_outputs
from src.util.MAPS import CRITERIONS_MAP, OPTIMIZER_MAP, SCHEDULER_MAP
from src.util.build import build_model_from_config, build_dataloader_from_config, build_criterion_from_config, \
    get_gpu_augmentations


class ClassificationModule(pl.LightningModule):
    """
    PyTorch Lightning Module for classification
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
                 gpu_augmentations: dict = None
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
        """
        super(ClassificationModule, self).__init__()
        self.model = build_model_from_config(model_conf)
        self.train_loader, self.val_loader = build_dataloader_from_config(loader_conf)
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
        for i in range(len(X)):
            X[i] = self.train_augs(X[i].squeeze())
        X = X.contiguous()
        out = self.model(X).squeeze(dim=1)
        loss = self.crit(out, y)
        return {'loss': loss, 'preds': torch.sigmoid(out).detach().cpu(), 'targets': y.detach().cpu()}

    def on_train_start(self) -> None:
        self.logger.log_hyperparams(self.hparams, {"hp/auroc": 0, "hp/auprc": 0})

    def training_epoch_end(self, outputs, key='train/'):
        y_pred, y_true, loss = _conc_outputs(outputs)
        try:
            metrics = classification_metrics(y_true, y_pred)
        except ValueError:
            tmp_shape = y_true.shape[0] + 1
            y_true_tmp = torch.zeros(tmp_shape)
            y_pred_tmp = torch.zeros(tmp_shape)
            y_true_tmp[:-1] = y_true
            y_pred_tmp[:-1] = y_pred
            y_true_tmp[-1] = 1
            y_pred_tmp[-1] = 1
            metrics = classification_metrics(y_true_tmp, y_pred_tmp)
        metrics['loss'] = loss.mean()
        mapped = {}
        for k in metrics.keys():
            mapped[key + k] = metrics[k]
        if key == "val/":
            mapped['hp/auroc'] = metrics['auroc']
            mapped['hp/auprc'] = metrics['auprc']
        del metrics
        self.log_dict(mapped, logger=True, on_epoch=True)

        if isinstance(self.train_loader.dataset, Slice2DDataset) and \
                'val' in key and \
                len(self.val_loader.dataset) == len(y_true):

            n_slices = len(self.train_loader.dataset.slice_indices)
            if len(y_true) % n_slices == 0:
                y_pred = y_pred.reshape(-1, n_slices).mean(dim=1)
                y_true = y_true.reshape(-1, n_slices).mean(dim=1)
                metrics = classification_metrics(y_true, y_pred)
                mapped = {}
                for k in metrics.keys():
                    mapped["per_study/" + k] = metrics[k]
                del metrics
                self.log_dict(mapped, logger=True, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        X, y = batch
        for i in range(len(X)):
            X[i] = self.val_augs(X[i].squeeze())
        X = X.contiguous()
        out = self.model(X).squeeze(dim=1)
        loss = self.crit(out, y)
        return {'loss': loss, 'preds': torch.sigmoid(out).detach().cpu(), 'targets': y.detach().cpu()}

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

    def predict_step(self, batch, batch_idx: int, dataloader_idx=None):
        X, y = batch
        for i in range(len(X)):
            X[i] = self.val_augs(X[i])
        out = self.model(X).squeeze(dim=1)
        return {'pred': torch.sigmoid(out), 'target': y}

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader
