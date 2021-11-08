import torch


def _conc_outputs(outputs):
    """
    Helper function concatenating outputs from a PyTorch lightning module
    during training
    :param outputs: Module outputs
    :return: float, float, float
    """
    y_pred = []
    y_true = []
    loss = []
    for pred in outputs:
        y_pred.append(pred['preds'])
        y_true.append(pred['targets'])
        loss.append(pred['loss'])

    y_pred = torch.hstack(y_pred)
    y_true = torch.hstack(y_true)
    loss = torch.stack(loss)
    return y_pred, y_true, loss