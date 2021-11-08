import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, auc, roc_auc_score, precision_recall_curve, \
    mean_squared_error, mean_absolute_error
import torch
from src.data.regression_transforms import BaseRegressionNormalizer


def classification_metrics(y_true: torch.Tensor,
                           y_pred: torch.Tensor
                           ):
    """
    Function computing classification metrics including
    Accuracy, Precision, Recall, F1-score, AUROC, AUPRC
    :param y_pred: (N,) predictions
    :param y_true: (N,) targets
    :return: dict[str,float]
    """
    # using (0,1) y_pred
    auroc = roc_auc_score(y_true, y_pred)
    prec, rec, _ = precision_recall_curve(y_true, y_pred)
    auprc = auc(rec, prec)
    # using {0,1} y_pred
    y_pred_bin = torch.zeros_like(y_pred)
    y_pred_bin[y_pred > 0.5] = 1.0
    acc = accuracy_score(y_true, y_pred_bin)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred_bin, average='binary', zero_division=0)

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auroc': auroc,
        'auprc': auprc,
    }


def regression_metrics(y_true: torch.Tensor,
                       y_pred: torch.Tensor,
                       scalar: BaseRegressionNormalizer):
    """
    Function computing the regression metrics including:
    MAE,MSE,RMSE
    :param y_pred: (N,) predictions
    :param y_true: (N,) targets
    :param scalar: Normalizer used
    :return: dict[str, float]
    """

    y_true_unscaled = scalar.inverse(y_true)
    y_pred_unscaled = scalar.inverse(y_pred)

    mse = mean_squared_error(y_true_unscaled, y_pred_unscaled)
    mae = mean_absolute_error(y_true_unscaled, y_pred_unscaled)

    return {
        'mae': mae,
        'mse': mse,
        'rmse': np.sqrt(mse)
    }
