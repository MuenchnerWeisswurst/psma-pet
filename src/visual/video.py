import os

import numpy as np
import cv2


def RGB2BGR(img: np.ndarray):
    """
    Helper function applying BGR colormap to each channel of a image
    :param img: Input image
    :return:
    """
    for i in range(img.shape[0]):
        img[i, ...] = cv2.cvtColor(img[i, ...], cv2.COLOR_RGB2BGR)
    return img


def make_video(img: np.ndarray,
               out_path: str,
               axis: int = 1
               ):
    """
    Function that converts a 3D image into a .mp4 video
    :param img: Input image
    :param out_path: Output file name
    :param axis: Axis to go through
    :return: Path to output file
    """
    os.makedirs(os.path.join(*out_path.split("/")[:-1]), exist_ok=True)
    x, y, img_slice = None, None, None
    if axis == 0:
        x, y = img.shape[1], img.shape[2]
    elif axis == 1:
        x, y = img.shape[0], img.shape[2]
    elif axis == 2:
        x, y = img.shape[0], img.shape[1]
    video_writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 10.0,
                                   (y, x), img.shape[-1] == 3)
    for i in range(img.shape[axis]):
        if axis == 0:
            img_slice = img[i, :, :]
        elif axis == 1:
            img_slice = img[:, i, :]
        elif axis == 2:
            img_slice = img[:, :, i]
        video_writer.write(img_slice)
    video_writer.release()
    return out_path


def norm_to_uint8(img: np.ndarray):
    """
    Helper function that converts an array to np.uint8
    :param img: Input array
    :return: Converted array
    """
    img_max, img_min = np.max(img), np.min(img)
    a = 255 / (img_max - img_min)
    b = 255 - a * img_max
    new_img = a * img + b
    return new_img.astype(np.uint8)