from typing import Tuple

import torchio as tio
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize

import src.data.regression_transforms as rtf
import src.data.transforms as tf
from src.data.dataset import PSMAPETDataset, InMemoryPSMAPETDataset, AugmentationDataset, PSMAPETCTDataset, \
    Slice2DDataset, MIPDataset
from src.models.gaussian_process import FeatureExtractor, SVDKLGP
from src.models.model_zoo import ResNet3D, DenseNet3D, DenseNet3DWithScalars, DenseNet2D
from src.util.MAPS import *


def get_gpu_augmentations(augmentations: dict):
    """
    Function transforming GPU augmentation config into a Compose object
    :param augmentations: Augmentation config
    :return: Compose
    """
    augments = [tf.TensorToCupy()]
    for aug in augmentations.keys():
        if aug == 'affine':
            augments.append(tf.RandomAffine(**augmentations['affine']))
        elif aug == 'elastic':
            augments.append(tf.RandomElasticDeformation(**augmentations['elastic']))
        elif aug == 'gamma':
            augments.append(tf.RandomGamma(**augmentations['gamma']))
        elif aug == 'normalize':
            augments.append(tf.PerSampleNormalize())
        elif aug == 'blur':
            augments.append(tf.RandomGaussianBlur(**augmentations['blur']))
        elif aug == 'noise':
            augments.append(tf.RandomNoise(**augmentations['noise']))
        else:
            raise NotImplementedError(f"Unsupported augmentation {aug}")
    augments.append(tf.CupyToTensor())
    return Compose(augments)


def get_cpu_augmentations(augmentations: dict):
    """
    Function transforming CPU augmentation config into a Compose object
    :param augmentations: Augmentation config
    :return: Compose
    """
    augments = []
    for aug in augmentations.keys():
        if aug == 'affine':
            augments.append(tio.RandomAffine(**augmentations['affine']))
        elif aug == 'elastic':
            augments.append(tio.RandomElasticDeformation(**augmentations['elastic']))
        elif aug == 'gamma':
            augments.append(tio.RandomGamma(**augmentations['gamma']))
        elif aug == 'normalize':
            augments.append(tio.PerSampleNormalize())
        elif aug == 'blur':
            augments.append(tio.RandomGaussianBlur(**augmentations['blur']))
        elif aug == 'noise':
            augments.append(tio.RandomNoise(**augmentations['noise']))
        else:
            raise NotImplementedError(f"Unsupported augmentation {aug}")
    return tio.Compose(augments)


def build_model_from_config(config: dict):
    """
    Function parsing a model config into a actual model
    :param config: Model configuration
    :return: Model
    """
    if config['name'] == 'resnet3d':
        n_blocks = config['n_blocks']
        residuals_per_block = config['residuals_per_block']
        channels_per_block = config['channels_per_block']
        batchnorm = config['batchnorm']
        dropout = config['dropout']
        activation = ACTIVATIONS_MAP[config['activation']]
        model = ResNet3D(
            n_blocks,
            residuals_per_block,
            channels_per_block,
            batchnorm=batchnorm,
            dropout=dropout,
            activation=activation,
            in_channels=config.get('in_channels', 1)
        )
    elif config['name'] == 'densenet3d':
        n_blocks = config['n_blocks']
        num_channels = config['num_channels']
        growth_rate = config['growth_rate']
        num_convs_per_block = config['channels_per_block']
        activation = ACTIVATIONS_MAP[config['activation']]
        kernel_size = (3, 3, 3) if 'kernel_size' not in config.keys() else config['kernel_size']
        padding = (1, 1, 1) if 'padding' not in config.keys() else config['padding']
        glob_pooling = torch.nn.AdaptiveMaxPool3d if 'glob_pooling' not in config.keys() else torch.nn.AdaptiveAvgPool3d
        model = DenseNet3D(
            num_channels,
            growth_rate,
            num_convs_per_block,
            activation=activation,
            kernel_size=kernel_size,
            padding=padding,
            glob_pooling=glob_pooling,
            in_channels=config.get('in_channels', 1)
        )
    elif config['name'] == 'densenet2d':
        n_blocks = config['n_blocks']
        num_channels = config['num_channels']
        growth_rate = config['growth_rate']
        num_convs_per_block = config['channels_per_block']
        activation = ACTIVATIONS_MAP[config['activation']]
        kernel_size = (3, 3) if 'kernel_size' not in config.keys() else config['kernel_size']
        padding = (1, 1) if 'padding' not in config.keys() else config['padding']
        glob_pooling = torch.nn.AdaptiveMaxPool2d if 'glob_pooling' not in config.keys() else torch.nn.AdaptiveAvgPool2d
        model = DenseNet2D(
            num_channels,
            growth_rate,
            num_convs_per_block,
            activation=activation,
            kernel_size=kernel_size,
            padding=padding,
            glob_pooling=glob_pooling,
            in_channels=config.get('in_channels', 1)
        )
    elif config['name'] == 'densenet3dwithscalers':
        n_blocks = config['n_blocks']
        num_channels = config['num_channels']
        growth_rate = config['growth_rate']
        num_convs_per_block = config['channels_per_block']
        activation = ACTIVATIONS_MAP[config['activation']]
        kernel_size = (3, 3, 3) if 'kernel_size' not in config.keys() else config['kernel_size']
        padding = (1, 1, 1) if 'padding' not in config.keys() else config['padding']
        glob_pooling = torch.nn.AdaptiveMaxPool3d if 'glob_pooling' not in config.keys() else torch.nn.AdaptiveAvgPool3d
        scalar_hidden_dims = config['scalar_hidden_dims']
        classifier_hidden_dims = config['classifier_hidden_dims']
        model = DenseNet3DWithScalars(
            num_channels,
            growth_rate,
            num_convs_per_block,
            scalar_hidden_dims,
            classifier_hidden_dims,
            activation=activation,
            kernel_size=kernel_size,
            padding=padding,
            glob_pooling=glob_pooling,
            in_channels=config.get('in_channels', 1)
        )
    elif config['name'] == 'pretrained_densenet121':
        model = models.densenet121(pretrained=True, progress=True)
        model.features[0] = torch.nn.Conv2d(1, 64, (7, 7), (2, 2), (3, 3), bias=False)
        model.classifier = torch.nn.Linear(1024, 1, bias=True)
    else:
        raise NotImplementedError(f"Model {config['name']} not supported!")

    return model


def build_gp_model_from_config(config: dict):
    nn = build_model_from_config(config)
    if config.get('pretrained_weights', None):
        nn.load_state_dict(torch.load(config['pretrained_weights'], map_location='cpu'))
    feature_extractor = FeatureExtractor(nn.net)
    if config['inducing_points'] == 'pretrained':
        return feature_extractor
    else:
        N = int(config['inducing_points'])
        inducing_points = torch.rand((N, feature_extractor.num_features)) * 2 - 1
        gpmodel = SVDKLGP(inducing_points, feature_extractor)
        return gpmodel


def build_dataloader_from_config(config: dict):
    """
    Function that parses a DataLoader config into a train and val DataLoader
    :param config: DataLoader config
    :return: train,val DataLoder
    """
    split = config['split']
    data_dir = config['data_dir']
    metadatafile = config['metadatafile']
    batch_size = config['batch_size']
    n_workers = config['n_workers']
    ct_data_dir = config.get('ct_data_dir', None)
    is_mip = config.get('MIP', False)
    transforms = []
    for tr in config['transforms']:
        if tr == 'totensor':
            transforms.append(tf.ToTensor())
        elif tr == 'normalize':
            if is_mip:
                transforms.append(Normalize(mean=[2245], std=[8348]))
            elif not ct_data_dir:
                transforms.append(Normalize(mean=[253], std=[1759]))
            else:
                transforms.append(tf.Normalize3D(mean=[253, -885],
                                                 std=[1759, 477]))

        elif tr == 'tonumpy':
            transforms.append(tf.ToNumpy())
        else:
            raise NotImplementedError(f"Transform {tr} not implemented")
    transforms = Compose(transforms)
    target_fun = TARGETS_MAP[config['target_fun']]
    in_memory = config['in_memory'] if 'in_memory' in config.keys() else None
    seed = config['seed'] if 'seed' in config.keys() else 1337
    augmentations = get_cpu_augmentations(config['cpu_augmentations']) if 'cpu_augmentations' in config.keys() else None
    size = config['size'] if 'size' in config.keys() else (256, 256, 263)
    use_pre_psa = config.get('use_pre_psa', False)
    slice_dataset_conf = config.get('slices', None)
    return get_train_val_split(split, data_dir, ct_data_dir, metadatafile, transforms, target_fun, augmentations,
                               batch_size,
                               n_workers, in_memory,
                               seed, size, use_pre_psa, slice_dataset_conf, is_mip)


def get_train_val_split(split: Tuple[int, int],
                        data_dir: str,
                        ct_data_dir: str,
                        metadatafile: str,
                        transforms: Compose,
                        target_fun,
                        augmentations: tio.Compose = None,
                        batch_size: int = 8,
                        n_workers: int = 8,
                        in_memory: str = None,
                        seed: int = 1337,
                        size: Tuple[int, int, int] = (256, 256, 263),
                        use_pre_psa: bool = False,
                        slice_conf: dict = None,
                        is_mip: bool = False,
                        ):
    """
    Function that creates the train and val DataLoader
    :param split: Train,val split ratio
    :param data_dir: Path to PET/CT data
    :param ct_data_dir: Path to CT data dir
    :param metadatafile: Path to metadatafile
    :param transforms: List of transforms to be applied
    :param target_fun: Function giving the target values
    :param augmentations: CPU augmentations to be applied
    :param batch_size: Batch size
    :param n_workers: Number of DataLoader workers
    :param in_memory: Whether the dataset should be preloaded to CPU/GPU
    :param seed: Random seed to dataset split
    :param size: Size of input images
    :param use_pre_psa: Whether pre PSA should be returned by the dataset
    :param slice_conf: Slice configuration
    :param is_mip: Whether MIP images should be returned
    :return: train,val DataLoader
    """
    if not ct_data_dir:
        dataset = PSMAPETDataset(data_dir, metadatafile, target_fun, data_transforms=transforms, size=size,
                                 use_pre_psa=use_pre_psa, mip=is_mip)
    else:
        dataset = PSMAPETCTDataset(data_dir, ct_data_dir, metadatafile, target_fun, data_transforms=transforms,
                                   size=size, use_pre_psa=use_pre_psa)

    if in_memory is None:
        pass
    elif in_memory == 'cpu':
        dataset = InMemoryPSMAPETDataset(dataset, device='cpu')
    elif in_memory == 'cuda:0':
        dataset = InMemoryPSMAPETDataset(dataset, device='cuda:0')
    else:
        raise NotImplementedError(f"Memory type {in_memory} unkown!")

    N = len(dataset)
    n_train = int(split[0] * N)
    n_val = N - n_train
    train_set, val_set = torch.utils.data.random_split(dataset, (n_train, n_val),
                                                       generator=torch.Generator().manual_seed(seed))

    if augmentations:
        train_set = AugmentationDataset(train_set, augmentations)
    elif slice_conf:
        axis = slice_conf['axis']
        slice_indices = slice_conf.get('slice_indices', None)
        if isinstance(slice_indices, dict):
            slice_indices = list(range(slice_indices['start'], slice_indices['stop']))

        train_set, val_set = Slice2DDataset(train_set, axis=axis, slice_indices=slice_indices), Slice2DDataset(val_set,
                                                                                                               axis=axis,
                                                                                                               slice_indices=slice_indices)
    elif is_mip:
        train_set = MIPDataset(train_set)
        val_set = MIPDataset(val_set)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=n_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=n_workers, pin_memory=True)

    return train_loader, val_loader


def build_criterion_from_config(crit,
                                crit_kwargs: dict,
                                train_loader: torch.utils.data.DataLoader
                                ):
    """
    Functions that builds the loss function
    :param crit: Loss function
    :param crit_kwargs: Arguments for the loss function
    :param train_loader: Train DataLoader
    :return: Criterion
    """
    args = {}

    def _get_targets(dataset) -> dict:
        if isinstance(dataset, PSMAPETDataset):
            return dataset.targets
        else:
            return _get_targets(dataset.dataset)

    for k, v in crit_kwargs.items():
        if k == 'pos_weights':
            if v is None:
                ts = _get_targets(train_loader.dataset)
                ys = []
                for pid, sidd in ts.items():
                    for sid in sidd.keys():
                        ys.append(ts[pid][sid])
                ys = torch.Tensor(ys)
                n_pos = torch.sum(ys == 1)
                n_neg = torch.sum(ys == 0)
                args[k] = torch.Tensor([n_neg / n_pos])
            else:
                args[k] = torch.Tensor([v])

    return crit(**args)


def build_regression_normalizer(type: str, config: dict):
    """
    Function parsing the regression target normalizer
    :param type: Type of normalization
    :param config: Normalizer config
    :return:
    """
    def _get_targ(config: dict):
        dataset = PSMAPETDataset(config['data_dir'], config['metadatafile'], TARGETS_MAP[config['target_fun']])
        targ = torch.empty(len(dataset))
        for i in range(len(targ)):
            s = dataset.studies[i]
            match = dataset.patient_regex.match(s)
            pid, sid = match.group(1), match.group(2)
            targ[i] = torch.as_tensor(dataset.targets[pid][sid])
        return targ

    if type == 'log_scale':
        targ = _get_targ(config)
        return rtf.LogScaleNormalizer(targ)
    elif type == 'log_scale_no_inverse':
        targ = _get_targ(config)
        return rtf.LogScaleNormalizerNoInverse(targ)
    elif type == 'log':
        return rtf.LogNormalizer()
    elif type == 'log_no_inverse':
        return rtf.LogNormalizerNoInverse()
    elif type == 'linear':
        return rtf.LinearNormalizer()
    elif type == 'mean_std':
        targ = _get_targ(config)
        return rtf.MeanStdNormalizer(targ)
    else:
        raise NotImplementedError(f"Unsupported normalizer type : {type}")


def get_testloader(config, host="/", in_mem=None, n_workers=4, batch_size=None):
    """
    Helper function returning the test DataLoader
    :param config: Whole configuration dict
    :param host: Prefix for the PET/CT (CT) data path
    :param in_mem: Whether the dataset should be preloaded
    :param n_workers: Number of DataLoader workers
    :param batch_size: Batch size
    :return: Test DataLoader
    """
    config['train_loader']['data_dir'] = host + config['train_loader']['data_dir']
    config['train_loader'][
        'metadatafile'] = host + "data/psma/test_set.csv"  # "/" + config['train_loader']['metadatafile']
    config['train_loader']['split'] = (1, 0)
    config['train_loader']['in_memory'] = in_mem
    config['train_loader']['batch_size'] = batch_size if batch_size else config['train_loader']['batch_size']
    config['train_loader']['seed'] = 1337
    if config['train_loader'].get('ct_data_dir', None):
        config['train_loader']['ct_data_dir'] = host + config['train_loader']['ct_data_dir']
    from_config = build_dataloader_from_config
    t, _ = from_config(config['train_loader'])
    test_loader = DataLoader(t.dataset, batch_size=config['train_loader']['batch_size'], num_workers=n_workers,
                             pin_memory=True)
    return test_loader
