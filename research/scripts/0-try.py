#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2022 Mar 07, 09:44:34
@last modified : 2022 Mar 07, 10:32:33
"""

import time

import torch
from sklearn.svm import LinearSVC
from torchvision.datasets import STL10

DATA_DIR = "../data/"

# list available compressors. b01 compresses the most (b01 > b005 > b001)
torch.hub.list("YannDubs/lossyless:main")
# ['clip_compressor_b001', 'clip_compressor_b005', 'clip_compressor_b01']

# Load the desired compressor and transformation to apply to images (by default on GPU if available)
compressor, transform = torch.hub.load(
    "YannDubs/lossyless:main", "clip_compressor_b005"
)

# Load some data to compress and apply transformation
stl10_train = STL10(DATA_DIR, download=True, split="train", transform=transform)
stl10_test = STL10(DATA_DIR, download=True, split="test", transform=transform)

# Compresses the datasets and save them to file (this requires GPU)
# Rate: 1506.50 bits/img | Encoding: 347.82 img/sec
compressor.compress_dataset(
    stl10_train,
    f"{DATA_DIR}/stl10_train_Z.bin",
    label_file=f"{DATA_DIR}/stl10_train_Y.npy",
)
compressor.compress_dataset(
    stl10_test,
    f"{DATA_DIR}/stl10_test_Z.bin",
    label_file=f"{DATA_DIR}/stl10_test_Y.npy",
)

# Load and decompress the datasets from file the datasets (does not require GPU)
# Decoding: 1062.38 img/sec
Z_train, Y_train = compressor.decompress_dataset(
    f"{DATA_DIR}/stl10_train_Z.bin", label_file=f"{DATA_DIR}/stl10_train_Y.npy"
)
Z_test, Y_test = compressor.decompress_dataset(
    f"{DATA_DIR}/stl10_test_Z.bin", label_file=f"{DATA_DIR}/stl10_test_Y.npy"
)

# Downstream STL10 evaluation. Accuracy: 98.65% | Training time: 0.5 sec
clf = LinearSVC(C=7e-3)
start = time.time()
clf.fit(Z_train, Y_train)
delta_time = time.time() - start
acc = clf.score(Z_test, Y_test)
print(
    f"Downstream STL10 accuracy: {acc*100:.2f}%.  \t Training time: {delta_time:.1f} "
)
