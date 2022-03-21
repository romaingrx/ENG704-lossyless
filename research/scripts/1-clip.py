#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2022 Mar 09, 14:43:15
@last modified : 2022 Mar 09, 15:45:13
"""

import torch

model_path = "../models/ViT-B-32.pt"

model = torch.jit.load(model_path).eval()
print(model)
