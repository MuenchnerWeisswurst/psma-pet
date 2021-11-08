import numpy as np
import pandas as pd

from src.data.util import group_frame_by_patients, dict_to_float


def psa_pre_post_relation(frame: pd.DataFrame):
    """
    Function computing PSA quotient for every patient and each study
    :param frame: Metadatafile dataframe
    :return: dict[int, dict[int, float]]
    """
    pre = dict_to_float(group_frame_by_patients(frame, value_key="PSA_pre_therapy"))
    post = dict_to_float(group_frame_by_patients(frame, value_key="PSA_post_therapy"))
    relation_dict = {}
    for patient_id in post.keys():
        if patient_id not in relation_dict.keys():
            relation_dict[patient_id] = {}
        for study_id in post[patient_id].keys():
            relation_dict[patient_id][study_id] = post[patient_id][study_id] / pre[patient_id][study_id]
    return relation_dict


def psa_post(frame: pd.DataFrame):
    """
    Function computing post PSA for every patient and each study
    :param frame:
    :return: dict[int, dict[int, float]]
    """
    post = dict_to_float(group_frame_by_patients(frame, value_key="PSA_post_therapy"))
    return post


def therapy_responsiveness(frame: pd.DataFrame):
    """
    Function computing the therapy responsiveness (q < 0.5) for every patient and each study
    :param frame: Metadatafile frame
    :return: dict[int, dict[int, float]]
    """
    relation_dict = psa_pre_post_relation(frame)
    for p_id in relation_dict.keys():
        for s_id in relation_dict[p_id].keys():
            relation_dict[p_id][s_id] = np.float32(1.0) if relation_dict[p_id][s_id] < 0.5 else np.float32(0.0)
    return relation_dict


def psa_pre(frame: pd.DataFrame):
    """
    Function computing post PSA for every patient and each study
    :param frame:
    :return: dict[int, dict[int, float]]
    """
    pre = dict_to_float(group_frame_by_patients(frame, value_key="PSA_pre_therapy"))
    return pre
