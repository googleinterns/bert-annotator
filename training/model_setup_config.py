#!/usr/bin/env python3
# Copyright 2020 Google LLC
#
# Licensed under the the Apache License v2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Details about the used model."""

from enum import Enum

ModelSize = Enum("ModelSize", "TINY BASE")


class ModelSetupConfig:
    """Stores details about the model setup."""
    def __init__(self,
                 size=None,
                 case_sensitive=False,
                 pretrained=True,
                 train_with_additional_labels=False):
        if not pretrained and size == ModelSize.BASE:
            raise ValueError(
                "Training from scratch is only supported for the tiny model.")
        if case_sensitive and size == ModelSize.TINY:
            raise ValueError(
                "Case sensitivity is only supported for the base model.")

        self.size = size
        self.case_sensitive = case_sensitive
        self.pretrained = pretrained
        self.train_with_additional_labels = train_with_additional_labels
