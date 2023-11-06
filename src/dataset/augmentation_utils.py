# code in this file is adpated from rpmcruz/autoaugment
# https://github.com/rpmcruz/autoaugment/blob/master/transformations.py
from __future__ import absolute_import, division, print_function

import random
from collections import defaultdict

import numpy as np
import PIL
import PIL.ImageDraw
import PIL.ImageEnhance
import PIL.ImageOps
import torch
from torchvision.transforms.transforms import Compose

random_mirror = True


def ShearX(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random_mirror and random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random_mirror and random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def TranslateX(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random_mirror and random.random() > 0.5:
        v = -v
    v = v * img.size[0]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random_mirror and random.random() > 0.5:
        v = -v
    v = v * img.size[1]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def TranslateXAbs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v <= 10
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateYAbs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v <= 10
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def Rotate(img, v):  # [-30, 30]
    assert -30 <= v <= 30
    if random_mirror and random.random() > 0.5:
        v = -v
    return img.rotate(v)


def AutoContrast(img, _):
    return PIL.ImageOps.autocontrast(img)


def Invert(img, _):
    return PIL.ImageOps.invert(img)


def Equalize(img, _):
    return PIL.ImageOps.equalize(img)


def Flip(img, _):  # not from the paper
    return PIL.ImageOps.mirror(img)


def Solarize(img, v):  # [0, 256]
    assert 0 <= v <= 256
    return PIL.ImageOps.solarize(img, v)


def Posterize(img, v):  # [4, 8]
    assert 4 <= v <= 8
    v = int(v)
    return PIL.ImageOps.posterize(img, v)


def Posterize2(img, v):  # [0, 4]
    assert 0 <= v <= 4
    v = int(v)
    return PIL.ImageOps.posterize(img, v)


def Contrast(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Color(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Color(img).enhance(v)


def Brightness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Sharpness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def Cutout(img, v):  # [0, 60] => percentage: [0, 0.2]
    assert 0.0 <= v <= 0.2
    if v <= 0.0:
        return img

    v = v * img.size[0]
    return CutoutAbs(img, v)


def CutoutAbs(img, v):  # [0, 60] => percentage: [0, 0.2]
    # assert 0 <= v <= 20
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.0))
    y0 = int(max(0, y0 - v / 2.0))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    # color = (0, 0, 0)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


def SamplePairing(imgs):  # [0, 0.4]
    def f(img1, v):
        i = np.random.choice(len(imgs))
        img2 = PIL.Image.fromarray(imgs[i])
        return PIL.Image.blend(img1, img2, v)

    return f


def augment_list(for_autoaug=True):  # 16 oeprations and their ranges
    l = [
        (ShearX, -0.3, 0.3),  # 0
        (ShearY, -0.3, 0.3),  # 1
        (TranslateX, -0.45, 0.45),  # 2
        (TranslateY, -0.45, 0.45),  # 3
        (Rotate, -30, 30),  # 4
        (AutoContrast, 0, 1),  # 5
        (Invert, 0, 1),  # 6
        (Equalize, 0, 1),  # 7
        (Solarize, 0, 256),  # 8
        (Posterize, 4, 8),  # 9
        (Contrast, 0.1, 1.9),  # 10
        (Color, 0.1, 1.9),  # 11
        (Brightness, 0.1, 1.9),  # 12
        (Sharpness, 0.1, 1.9),  # 13
        (Cutout, 0, 0.2),  # 14
        # (SamplePairing(imgs), 0, 0.4),  # 15
    ]
    if for_autoaug:
        l += [
            (CutoutAbs, 0, 20),  # compatible with auto-augment
            (Posterize2, 0, 4),  # 9
            (TranslateXAbs, 0, 10),  # 9
            (TranslateYAbs, 0, 10),  # 9
        ]
    return l


augment_dict = {fn.__name__: (fn, v1, v2) for fn, v1, v2 in augment_list()}


def get_augment(name):
    return augment_dict[name]


def apply_augment(img, name, level):
    augment_fn, low, high = get_augment(name)
    return augment_fn(img.copy(), level * (high - low) + low)


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = torch.Tensor(eigval)
        self.eigvec = torch.Tensor(eigvec)

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = (
            self.eigvec.type_as(img)
            .clone()
            .mul(alpha.view(1, 3).expand(3, 3))
            .mul(self.eigval.view(1, 3).expand(3, 3))
            .sum(1)
            .squeeze()
        )

        return img.add(rgb.view(3, 1, 1).expand_as(img))


PARAMETER_MAX = 10


def float_parameter(level, maxval):
    return float(level) * maxval / PARAMETER_MAX


def int_parameter(level, maxval):
    return int(float_parameter(level, maxval))


def autoaug2arsaug(f):
    def autoaug():
        mapper = defaultdict(lambda: lambda x: x)
        mapper.update(
            {
                "ShearX": lambda x: float_parameter(x, 0.3),
                "ShearY": lambda x: float_parameter(x, 0.3),
                "TranslateX": lambda x: int_parameter(x, 10),
                "TranslateY": lambda x: int_parameter(x, 10),
                "Rotate": lambda x: int_parameter(x, 30),
                "Solarize": lambda x: 256 - int_parameter(x, 256),
                "Posterize2": lambda x: 4 - int_parameter(x, 4),
                "Contrast": lambda x: float_parameter(x, 1.8) + 0.1,
                "Color": lambda x: float_parameter(x, 1.8) + 0.1,
                "Brightness": lambda x: float_parameter(x, 1.8) + 0.1,
                "Sharpness": lambda x: float_parameter(x, 1.8) + 0.1,
                "CutoutAbs": lambda x: int_parameter(x, 20),
            }
        )

        def low_high(name, prev_value):
            _, low, high = get_augment(name)
            return float(prev_value - low) / (high - low)

        policies = f()
        new_policies = []
        for policy in policies:
            new_policies.append(
                [
                    (name, pr, low_high(name, mapper[name](level)))
                    for name, pr, level in policy
                ]
            )
        return new_policies

    return autoaug


@autoaug2arsaug
def autoaug_paper_cifar10():
    return [
        [("Invert", 0.1, 7), ("Contrast", 0.2, 6)],
        [("Rotate", 0.7, 2), ("TranslateXAbs", 0.3, 9)],
        [("Sharpness", 0.8, 1), ("Sharpness", 0.9, 3)],
        [("ShearY", 0.5, 8), ("TranslateYAbs", 0.7, 9)],
        [("AutoContrast", 0.5, 8), ("Equalize", 0.9, 2)],
        [("ShearY", 0.2, 7), ("Posterize2", 0.3, 7)],
        [("Color", 0.4, 3), ("Brightness", 0.6, 7)],
        [("Sharpness", 0.3, 9), ("Brightness", 0.7, 9)],
        [("Equalize", 0.6, 5), ("Equalize", 0.5, 1)],
        [("Contrast", 0.6, 7), ("Sharpness", 0.6, 5)],
        [("Color", 0.7, 7), ("TranslateXAbs", 0.5, 8)],
        [("Equalize", 0.3, 7), ("AutoContrast", 0.4, 8)],
        [("TranslateYAbs", 0.4, 3), ("Sharpness", 0.2, 6)],
        [("Brightness", 0.9, 6), ("Color", 0.2, 6)],
        [("Solarize", 0.5, 2), ("Invert", 0.0, 3)],
        [("Equalize", 0.2, 0), ("AutoContrast", 0.6, 0)],
        [("Equalize", 0.2, 8), ("Equalize", 0.6, 4)],
        [("Color", 0.9, 9), ("Equalize", 0.6, 6)],
        [("AutoContrast", 0.8, 4), ("Solarize", 0.2, 8)],
        [("Brightness", 0.1, 3), ("Color", 0.7, 0)],
        [("Solarize", 0.4, 5), ("AutoContrast", 0.9, 3)],
        [("TranslateYAbs", 0.9, 9), ("TranslateYAbs", 0.7, 9)],
        [("AutoContrast", 0.9, 2), ("Solarize", 0.8, 3)],
        [("Equalize", 0.8, 8), ("Invert", 0.1, 3)],
        [("TranslateYAbs", 0.7, 9), ("AutoContrast", 0.9, 1)],
    ]


class CutoutDefault(object):
    """
    Reference : https://github.com/quark0/darts/blob/master/cnn/utils.py
    """

    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1:y2, x1:x2] = 0.0
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


class Augmentation(object):
    def __init__(self, policies):
        self.policies = policies

    def __call__(self, img):
        for _ in range(1):
            policy = random.choice(self.policies)
            for name, pr, level in policy:
                if random.random() > pr:
                    continue
                img = apply_augment(img, name, level)
        return img
