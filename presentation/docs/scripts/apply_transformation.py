#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2022 Mar 16, 10:46:29
@last modified : 2022 Mar 16, 11:09:59
"""

from sys import argv
from functools import partial
import torchvision.transforms.functional as TF
from PIL import Image

transforms = dict(
    # rotation=partial(TF.rotate, angle=40),
    hflip=TF.hflip,
    saturation=partial(TF.adjust_saturation, saturation_factor=2),
    brightness=partial(TF.adjust_brightness, brightness_factor=2),
    gaussian=partial(TF.gaussian_blur, kernel_size=(5, 5)),
    gray=TF.to_grayscale,
)

from matplotlib.pyplot import imshow, show

image_path = argv[1]
*_, image_name = image_path.split("/")
base_name, extension = image_name.split(".")

output_dir = argv[2]
img = Image.open(image_path)

for trans, aug in transforms.items():
    output_filepath = f"{output_dir}{trans}.{extension}"
    augmented_img = aug(img)
    augmented_img.save(output_filepath)
