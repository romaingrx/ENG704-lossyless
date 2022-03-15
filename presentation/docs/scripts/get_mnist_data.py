#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2022 Mar 15, 17:03:56
@last modified : 2022 Mar 15, 17:34:18
"""

from torchvision.datasets import MNIST
import torchvision.transforms as T

DATA_DIR = "~/datasets/"

mnist = MNIST(DATA_DIR, download=True, train=True)


def get_desired_images(wanted_digits):
    imgs = []
    for img, label in mnist:
        if len(imgs) == len(wanted_digits):
            break

        if label == wanted_digits[len(imgs)]:
            imgs.append(img)
    return imgs


def transform(imgs):
    transforms = T.Compose(
        [
            T.RandomAffine(25, (0.25, 0.25), (1.25, 1.25)),
        ]
    )
    return [transforms(img) for img in imgs]


def save_to_file(dir, imgs, labels):
    for idx, (img, label) in enumerate(zip(imgs, labels)):
        img.save(dir + f"/{idx}-{label}.png")


wanted_digits = [0, 1, 1, 8, 8]
prototypicals = get_desired_images(wanted_digits)
sources = transform(prototypicals)

save_to_file("../imgs/mnist/source", sources, wanted_digits)
save_to_file("../imgs/mnist/prototypicals", prototypicals, wanted_digits)
