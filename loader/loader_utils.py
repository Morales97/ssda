import numpy as np
from PIL import Image


def _build_size(orig_img, width, height):
    size = [width, height]
    if size[0] == -1: size[0] = orig_img.width
    if size[1] == -1: size[1] = orig_img.height
    return size


# 3Gb / 300k = 10000 (per worker)
# @lru_cache(maxsize=5000)
def _load_lru_cache(*args, **kwargs):
    return _load(*args, **kwargs)


def _load(_path, is_segmentation, resize, width, height, convert_segmentation=True):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(_path, 'rb') as f:
        with Image.open(f) as _img:
            if is_segmentation:
                if convert_segmentation:
                    _img = _img.convert()
                if resize: _img = _img.resize(_build_size(_img, width, height), Image.NEAREST)
            else:
                _img = _img.convert('RGB')
                if resize: _img = _img.resize(_build_size(_img, width, height), Image.ANTIALIAS)
    # print(np.asarray(_img).nbytes/1e6)
    return _img


def pil_loader(path, std_width, std_height, is_segmentation=False, lru_cache=False, convert_segmentation=True):
    if lru_cache:
        load_fn = _load_lru_cache
    else:
        load_fn = _load
    return load_fn(path, is_segmentation, True, std_width, std_height, convert_segmentation=convert_segmentation)

