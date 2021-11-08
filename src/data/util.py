import os
import re
from typing import Union

import nibabel as nib
import numpy as np
import pandas as pd


def group_frame_by_patients(frame: pd.DataFrame,
                            key_key: str = "pseudonym_study",
                            value_key: str = "diff_exam_therapy_days"
                            ):
    """
    Helper function that transforms a pd.DataFrame into a dict
    :param frame: Metadatafile frame
    :param key_key: Frame column used as dict key
    :param value_key: Frame column used as dict value
    :return: dict[key, dict[key', value]]
    """
    # group by patient
    patient_regex = re.compile(r"psma_([0-9]+)_([0-9]+)")
    grouped_dict = {}
    frame = frame.dropna()
    for _, series in frame.iterrows():
        key = series[key_key]
        value = series[value_key]
        if match := patient_regex.match(key):
            patient_id = match.group(1)
            study_id = match.group(2)
            if patient_id not in grouped_dict.keys():
                grouped_dict[patient_id] = {}
            grouped_dict[patient_id][study_id] = value
        else:
            print(f"not match {key}")
    return grouped_dict


def map_to_first_study(d: dict):
    """
    Helper funtion that returns the first value of each value of a dict
    :param d: dict to be mapped
    :return: dict
    """
    return {p_id: study_dict['0'] for p_id, study_dict in d.items()}


def dict_to_float(d: dict):
    """
    Helper function that maps every value of a dict to float
    :param d: input dict
    :return: mapped dict
    """
    res = {}
    for p_id in d.keys():
        res[p_id] = {}
        for s_id in d[p_id].keys():
            res[p_id][s_id] = np.float32(float(
                d[p_id][s_id].replace(">", "").replace("<", "").replace(",", ".").strip()) if isinstance(d[p_id][s_id],
                                                                                                         str) else float(
                d[p_id][s_id]))
    return res


def load_file(base_dir: str,
              patient_idx: Union[int, str],
              study_idx: Union[int, str],
              examinination_idx: Union[int, str] = 0,
              size_key: str = "",
              mip_key: str = ""
              ):
    """
    Helper function that loads a nifty image into memory
    :param base_dir: Path to nifty images
    :param patient_idx: Patient ID
    :param study_idx: Study ID
    :param examinination_idx: Examination ID
    :param size_key: Preprocessed image to be loaded
    :param mip_key: Preprocessed MIP image to be loaded
    :return: np.ndarray
    """
    file_path = os.path.join(base_dir, f"psma_{patient_idx}", f"psma_{patient_idx}_{study_idx}",
                             f"psma_{patient_idx}_{study_idx}_{examinination_idx}{size_key}{mip_key}.nii.gz")
    img = nib.load(file_path)
    img = img.get_fdata().astype(np.float32)
    return img
