import re
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from tqdm import tqdm

from src.data.util import group_frame_by_patients, dict_to_float
from src.data.util import load_file


class PSMAPETDataset(Dataset):
    """
    Basic PSMA PET/CT dataset class
    """

    def __init__(self,
                 data_base_dir: str,
                 metadatafile: str,
                 compute_target_fun,
                 data_transforms: Compose = None,
                 target_transforms: Compose = None,
                 size: Tuple[int, int, int] = (256, 256, 263),
                 use_pre_psa: bool = False,
                 mip: bool = False
                 ):
        """

        :param data_base_dir: Path to nifty images
        :param metadatafile: Path to metadata csv
        :param compute_target_fun: Function computing target values (y)
        :param data_transforms: List of transforms to be applied to images
        :param target_transforms: List of transforms to be applied to the targets
        :param size: Shape of images to be used
        :param use_pre_psa: Whether pre PSA value should be returned
        :param mip: Whether MIP images should be returned
        """
        self.patient_regex = re.compile(r"psma_([0-9]+)_([0-9]+)")
        self.data_base_dir = data_base_dir
        self.metadatafile = metadatafile
        self.meta_frame = pd.read_csv(self.metadatafile).dropna()

        self.studies = sorted(list(set(self.meta_frame['pseudonym_study'].values)))
        self.targets = compute_target_fun(self.meta_frame)

        self.data_transforms = data_transforms
        self.target_transforms = target_transforms
        self.size_key = "x".join(str(o) for o in size) if size is not None else ""
        self.size_key = "_" + self.size_key if size is not None else ""
        self.use_pre_psa = use_pre_psa
        self.pre_psa = dict_to_float(group_frame_by_patients(self.meta_frame, value_key="PSA_pre_therapy"))
        self.mip_key = "_mip" if mip else ""

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, item):
        study = self.studies[item]
        match = self.patient_regex.match(study)
        patient_id, study_id = match.group(1), match.group(2)

        image = load_file(self.data_base_dir, patient_id, study_id, size_key=self.size_key, mip_key=self.mip_key)
        image = np.expand_dims(image, 0)
        target = self.targets[patient_id][study_id]

        if self.data_transforms is not None:
            image = self.data_transforms(image)
        if self.target_transforms is not None:
            target = self.target_transforms(target)

        if self.use_pre_psa:
            pre_psa = self.pre_psa[patient_id][study_id]
            image = {'image': image, 'psa': torch.as_tensor(pre_psa)}
        return image, target

    def get_target(self, item):
        """
        Method returning the corresponding target
        :param item: Index
        :return: target
        """
        study = self.studies[item]
        match = self.patient_regex.match(study)
        patient_id, study_id = match.group(1), match.group(2)
        target = self.targets[patient_id][study_id]

        if self.target_transforms is not None:
            target = self.target_transforms(target)

        return target


class PSMAPETCTDataset(Dataset):
    """
        Basic PSMA PET/CT+CT dataset class
    """

    def __init__(self,
                 pet_data_base_dir: str,
                 ct_data_base_dir: str,
                 metadatafile: str,
                 compute_target_fun,
                 data_transforms: Compose = None,
                 target_transforms: Compose = None,
                 size: Tuple[int, int, int] = (256, 256, 263),
                 use_pre_psa: bool = False
                 ):

        """

        :param pet_data_base_dir: Path to nifty PET/CT images
        :param ct_data_base_dir: Path to nifty CT images
        :param metadatafile: Path to metadata csv
        :param compute_target_fun: Function computing target values (y)
        :param data_transforms: List of transforms to be applied to images
        :param target_transforms: List of transforms to be applied to the targets
        :param size: Shape of images to be used
        :param use_pre_psa: Whether pre PSA value should be returned
        """
        self.patient_regex = re.compile(r"psma_([0-9]+)_([0-9]+)")
        self.pet_data_base_dir = pet_data_base_dir
        self.ct_data_base_dir = ct_data_base_dir
        self.metadatafile = metadatafile
        self.meta_frame = pd.read_csv(self.metadatafile).dropna()

        self.studies = sorted(list(set(self.meta_frame['pseudonym_study'].values)))
        self.targets = compute_target_fun(self.meta_frame)

        self.data_transforms = data_transforms
        self.target_transforms = target_transforms
        self.size_key = "x".join(str(o) for o in size) if size is not None else ""
        self.size_key = "_" + self.size_key if size is not None else ""
        self.use_pre_psa = use_pre_psa
        self.pre_psa = dict_to_float(group_frame_by_patients(self.meta_frame, value_key="PSA_pre_therapy"))

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, item):
        study = self.studies[item]
        match = self.patient_regex.match(study)
        patient_id, study_id = match.group(1), match.group(2)

        pet_image = load_file(self.pet_data_base_dir, patient_id, study_id, size_key=self.size_key)
        try:
            ct_image = load_file(self.ct_data_base_dir, patient_id, study_id, examinination_idx=1,
                                 size_key=self.size_key)
        except FileNotFoundError:
            ct_image = load_file(self.ct_data_base_dir, patient_id, study_id, examinination_idx=2,
                                 size_key=self.size_key)

        image = np.stack([pet_image, ct_image])
        target = self.targets[patient_id][study_id]

        if self.data_transforms is not None:
            image = self.data_transforms(image)
        if self.target_transforms is not None:
            target = self.target_transforms(target)

        if self.use_pre_psa:
            pre_psa = self.pre_psa[patient_id][study_id]
            image = {'image': image, 'psa': torch.as_tensor(pre_psa)}
        return image, target

    def get_target(self, item):
        study = self.studies[item]
        match = self.patient_regex.match(study)
        patient_id, study_id = match.group(1), match.group(2)
        target = self.targets[patient_id][study_id]

        if self.target_transforms is not None:
            target = self.target_transforms(target)

        return target


class InMemoryPSMAPETDataset(Dataset):
    """
    Helper class used for preloading a Dataset into memory
    """

    def __init__(self,
                 init_dataset: PSMAPETDataset,
                 device: str = "cpu"
                 ):
        """

        :param init_dataset: Dataset that should be preloaded
        :param device: Memory device where the dataset should be loaded to
        """
        self.dataset = init_dataset
        self.images = []
        self.target = []
        self.device = device
        print(f"Loading PSMADataset in memory...")
        for i in tqdm(range(len(self.dataset))):
            X, y = self.dataset[i]
            if isinstance(X, dict):
                for k, v in X.items():
                    X[k] = v.to(self.device)
            else:
                X = X.to(device)
            self.images.append(X)
            if not isinstance(y, torch.Tensor):
                y = torch.as_tensor(y)
            self.target.append(y.to(self.device))

        # self.images = torch.stack(self.images)
        # self.target = torch.stack(self.target).squeeze()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.images[item], self.target[item]


class Slice2DDataset(Dataset):
    """
    Helper class for a sliced dataset
    """

    def __init__(
            self,
            init_dataset: Union[InMemoryPSMAPETDataset, PSMAPETDataset],
            axis: int = 1,
            slice_indices: List[int] = None
    ):
        """

        :param init_dataset: Dataset where the images should be sliced
        :param axis: Axis which should be sliced
        :param slice_indices: Indices that should be used on :param axis:
        """
        self.dataset = init_dataset
        if axis > 2:
            raise Exception(f"Axis {axis} does not exist")
        self.axis = axis
        if not slice_indices and len(self.dataset) > 0:
            n = self.dataset[0][0].shape[self.axis]
            self.slice_indices = list(range(n))
        else:
            self.slice_indices = slice_indices

    def __len__(self):
        return len(self.dataset) * len(self.slice_indices)

    def __getitem__(self, item):
        dataset_idx = item // len(self.slice_indices)
        slice_idx = item % len(self.slice_indices)
        slice_idx = self.slice_indices[slice_idx]
        X, y = self.dataset[dataset_idx]
        if self.axis == 0:
            img_slice = X[:, slice_idx, :, :]
        elif self.axis == 1:
            img_slice = X[:, :, slice_idx, :]
        else:
            img_slice = X[:, :, :, slice_idx]
        return img_slice, y


class MIPDataset(Dataset):
    """
    Helper class for MIP image dataset
    """

    def __init__(self,
                 init_dataset: Union[PSMAPETDataset, InMemoryPSMAPETDataset]
                 ):
        """

        :param init_dataset: Dataset with MIP images
        """
        self.dataset = init_dataset
        if len(init_dataset) > 0:
            self.n_angles = init_dataset[0][0].shape[1]
        else:
            self.n_angles = 0

    def __len__(self):
        return len(self.dataset) * self.n_angles

    def __getitem__(self, item):
        dataset_idx = item // self.n_angles
        slice_idx = item % self.n_angles
        X, y = self.dataset[dataset_idx]
        return X[:, slice_idx, ...], y


class AugmentationDataset(Dataset):
    """
    Helper class for augmentations applied on a preloaded dataset
    """

    def __init__(self,
                 init_dataset: Union[PSMAPETDataset, InMemoryPSMAPETDataset],
                 augmentations: Compose = None
                 ):
        """

        :param init_dataset: Preloaded dataset
        :param augmentations: List of augmentations applied to images
        """
        self.dataset = init_dataset
        self.augmentations = augmentations

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        X, y = self.dataset[item]
        if self.augmentations:
            return self.augmentations(X.unsqueeze(0)).squeeze(), y
        return X, y
