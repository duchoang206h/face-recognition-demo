import PIL.Image
import io
import numpy as np


def load_image_contents(contents, mode='RGB'):
    im = PIL.Image.open(io.BytesIO(contents))
    if mode:
        im = im.convert(mode)
    return np.array(im)
