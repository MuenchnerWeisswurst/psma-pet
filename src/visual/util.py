import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize

from src.data.transforms import Resize3D
from src.models.classification import ClassificationModule
from src.util.build import build_model_from_config
from src.visual.video import norm_to_uint8


def create_superimposed_img(input_image: np.ndarray,
                            heatmap: np.ndarray,
                            colormap="jet",
                            weight: float = 0.2,
                            normed_threshold: int = None,
                            unnormed_threshold: float = None,
                            original_image_cmap=None,
                            ):
    """
    Function computing a superimposed image
    :param input_image: The input image
    :param heatmap: Heatmap to be added to the input image
    :param colormap: Colormap to be applied to the input image
    :param weight: Weight for the heatmap
    :param normed_threshold: Threshold only applied to values <
    :param unnormed_threshold: Threshold applied to values: -unnormed_threshold < values < unnormed_threshold
    :param original_image_cmap: Colormap applied to the original image
    :return: Superimposed image
    """
    if input_image.shape != heatmap.shape:
        resizer = Resize3D(out_shape=input_image.shape)
        heatmap_resize = resizer(heatmap)
        heatmap_resize = norm_to_uint8(heatmap_resize)
    else:
        heatmap_resize = norm_to_uint8(heatmap)
    idx = None
    orig_image_norm = Normalize(vmin=0, vmax=255, clip=True)
    orig_image_mappable = plt.cm.ScalarMappable(cmap=plt.get_cmap(original_image_cmap), norm=orig_image_norm)
    heatmap_mappable = plt.cm.ScalarMappable(cmap=plt.get_cmap(colormap), norm=orig_image_norm)
    normed_input = norm_to_uint8(input_image)
    superimposed_img = np.zeros((*normed_input.shape, 3), dtype=np.uint8)
    for i in range(normed_input.shape[0]):
        islice = heatmap_resize[i, ...]
        input_islice = normed_input[i, ...]
        if normed_threshold:
            idx = islice < normed_threshold  # 100
        elif unnormed_threshold:
            idx = np.logical_and(-unnormed_threshold < islice, islice < unnormed_threshold)
        islice = norm_to_uint8(heatmap_mappable.to_rgba(islice))
        islice = cv2.cvtColor(islice, cv2.COLOR_RGBA2RGB)

        islice[idx] = np.asarray([0, 0, 0])

        if not original_image_cmap:
            input_islice_rgb = np.zeros((*input_islice.shape, 3))
            for j in range(3):
                input_islice_rgb[..., j] = input_islice
        else:
            input_islice_rgb = orig_image_mappable.to_rgba(input_islice)
            input_islice_rgb = norm_to_uint8(input_islice_rgb)
            input_islice_rgb = cv2.cvtColor(input_islice_rgb, cv2.COLOR_RGBA2RGB)
        superimposed_img[i, ...] = islice * weight + input_islice_rgb

    superimposed_img = norm_to_uint8(superimposed_img)
    return superimposed_img


def get_model(ckpt_path, config, n_layers=7):
    """
    Function that parses a PyTorch lightning checkpoint and its config and returns the corresponding model
    with loaded weights
    :param ckpt_path: Path to PyTorch Lightning checkpoint
    :param config: Module config
    :param n_layers: Index of last layer to be returned
    :return: Model
    """
    if "default" in ckpt_path:
        module = ClassificationModule.load_from_checkpoint(
            ckpt_path,
            map_location="cpu",
            model_conf=config['model'],
            loader_conf=config['train_loader'],
            crit=config['crit'],
            crit_kwargs=config.get('crit_kwargs', None),
            optim=config['optim'],
            optim_kwargs=config['optim_kwargs'],
            scheduler=config.get('scheduler', None),
            scheduler_metric=config.get('scheduler_metric', None),
            scheduler_kwargs=config.get('scheduler_kwargs', None),
        )
        return module.model.net[:n_layers]
    else:
        module = build_model_from_config(config['model'])
        module.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        return module.net[:n_layers]
