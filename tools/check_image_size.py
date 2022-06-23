import numpy as np
import os
import xml.etree.ElementTree as ET
import cv2
from fvcore.common.file_io import PathManager
from PIL import Image

_EXIF_ORIENT = 274  # exif 'Orientation' tag

def _apply_exif_orientation(image):
    """
    Applies the exif orientation correctly.

    This code exists per the bug:
      https://github.com/python-pillow/Pillow/issues/3973
    with the function `ImageOps.exif_transpose`. The Pillow source raises errors with
    various methods, especially `tobytes`

    Function based on:
      https://github.com/wkentaro/labelme/blob/v4.5.4/labelme/utils/image.py#L59
      https://github.com/python-pillow/Pillow/blob/7.1.2/src/PIL/ImageOps.py#L527

    Args:
        image (PIL.Image): a PIL image

    Returns:
        (PIL.Image): the PIL image with exif orientation applied, if applicable
    """
    if not hasattr(image, "getexif"):
        return image

    exif = image.getexif()

    if exif is None:
        return image

    orientation = exif.get(_EXIF_ORIENT)

    method = {
        2: Image.FLIP_LEFT_RIGHT,
        3: Image.ROTATE_180,
        4: Image.FLIP_TOP_BOTTOM,
        5: Image.TRANSPOSE,
        6: Image.ROTATE_270,
        7: Image.TRANSVERSE,
        8: Image.ROTATE_90,
    }.get(orientation)

    if method is not None:
        return image.transpose(method)
    return image

# dataset_root = './datasets/clipart'
# anno_path = os.path.join(dataset_root, 'Annotations')
# image_path = os.path.join(dataset_root, 'JPEGImages')
# with PathManager.open(os.path.join(dataset_root, "ImageSets", "Main", "train.txt")) as f:
#     image_list = np.loadtxt(f, dtype=np.str)

# dataset_root = './datasets/comic'
# anno_path = os.path.join(dataset_root, 'Annotations')
# image_path = os.path.join(dataset_root, 'JPEGImages')
# with PathManager.open(os.path.join(dataset_root, "ImageSets", "Main", "all.txt")) as f:
#     image_list = np.loadtxt(f, dtype=np.str)

dataset_root = './datasets/watercolor'
anno_path = os.path.join(dataset_root, 'Annotations')
image_path = os.path.join(dataset_root, 'JPEGImages')
with PathManager.open(os.path.join(dataset_root, "ImageSets", "Main", "all.txt")) as f:
    image_list = np.loadtxt(f, dtype=np.str)

for name in image_list:    
    image = Image.open(os.path.join(image_path, name + ".jpg"))
    image = _apply_exif_orientation(image)
    image_width, image_height = image.size
    with PathManager.open(os.path.join(anno_path, name + ".xml")) as f:
        tree = ET.parse(f)
    anno_height = int(tree.findall("./size/height")[0].text)
    anno_width = int(tree.findall("./size/width")[0].text)
    if image_height != anno_height or image_width != anno_width:
        print('{}: img shape ({},{}), anno shape ({}, {})'.format(name, image_height, image_width, anno_height, anno_width))

