import os
from multiprocessing import Pool
from typing import List

import nibabel as nib
import numpy as np
from scipy.ndimage.interpolation import rotate
from tqdm import tqdm

from src.data.transforms import Resize3D

# Set nibal compression level to max compression
nib.openers.Opener.default_compresslevel = 9


def resize_img(in_file: str, out_file: str, resizer: Resize3D):
    """
    Function that loads an image, resizes it and writes it to disk
    :param in_file: Path to image to be resized
    :param out_file: Path to resized image
    :param resizer: Callable resizer object
    :return:
    """
    img = nib.load(in_file)
    aff = img.affine
    img = img.get_fdata().astype(np.float32)
    img = resizer(img)
    img = nib.Nifti1Image(img, aff)
    nib.save(img, out_file)


def mip(in_file: str, out_file: str, angles: List[int], axis: int = 1):
    """
    Function that loads an image, rotates it around a specific axis and angles and writes it to disk
    :param in_file: Path to image to be resized
    :param out_file: Path to resized image
    :param angles: List of angles in degrees
    :param axis: Rotation axis
    :return:
    """
    img = nib.load(in_file)
    aff = img.affine
    img = img.get_fdata().astype(np.float32)
    resizer = Resize3D((256, 256))
    mips = []
    for angle in angles:
        mip = rotate(img, angle)
        mip = np.amax(mip, axis=axis)
        mip = resizer(mip)
        mips.append(mip)
    img = np.stack(mips)
    img = nib.Nifti1Image(img, aff)
    nib.save(img, out_file)


def preprocess_resize(base_datadir, out_size):
    """
    Function that applies resize_img to every nifty image found in base_datadir
    asynchronously
    :param base_datadir: Path to nifty images
    :param out_size: Target size of images
    :return:
    """
    in_file_paths = []

    for patient in os.listdir(base_datadir):
        patient_dir = os.path.join(base_datadir, patient)
        for study in os.listdir(patient_dir):
            study_dir = os.path.join(patient_dir, study)
            for file in os.listdir(study_dir):
                file_path = os.path.join(study_dir, file)
                if "x" not in file_path:
                    in_file_paths.append(file_path)

    out_file_paths = list(
        map(
            lambda x: os.path.join("/", *x.split("/")[:-1], x.split("/")[-1].replace(".nii.gz",
                                                                                     f"_{'x'.join(str(o) for o in out_size)}.nii.gz")),
            in_file_paths
        )
    )

    pool = Pool(128)
    res = []
    for i, o in zip(in_file_paths, out_file_paths):
        resizer = Resize3D(out_shape=out_size)
        res.append(pool.apply_async(resize_img, (i, o, resizer)))

    for r in tqdm(res):
        r.get()


def preprocess_mip(base_datadir, angles=None):
    """
    Function that applies mip to every nifty image found in base_datadir
    asynchronously
    :param base_datadir: Path to nifty images
    :param angles: Angles to be used
    :return:
    """
    if not angles:
        angles = list(range(0, 360, 10))

    in_file_paths = []
    for patient in os.listdir(base_datadir):
        patient_dir = os.path.join(base_datadir, patient)
        for study in os.listdir(patient_dir):
            study_dir = os.path.join(patient_dir, study)
            for file in os.listdir(study_dir):
                file_path = os.path.join(study_dir, file)
                if "256x" in file_path:
                    in_file_paths.append(file_path)

    pool = Pool(128)
    res = []
    for x in in_file_paths:
        out_file = os.path.join(*x.split("/")[:-1], x.split("/")[-1].replace(".nii.gz", f"_mip.nii.gz"))
        res.append(pool.apply_async(mip, (x, out_file, angles)))

    for r in tqdm(res):
        r.get()
