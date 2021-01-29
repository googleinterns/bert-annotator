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

from dataclasses import dataclass
from enum import Enum

ModelSize = Enum("ModelSize", "TINY BASE")


@dataclass
class ModelSetupConfig:
    """Stores details about the model setup."""
    size: ModelSize = ModelSize.BASE
    case_sensitive: bool = False
    pretrained: bool = True
    train_with_additional_labels: bool = False

    def __post_init__(self):
        if not self.pretrained and self.size == ModelSize.BASE:
            raise ValueError(
                "Training from scratch is only supported for the tiny model.")
        if self.case_sensitive and self.size == ModelSize.TINY:
            raise ValueError(
                "Case sensitivity is only supported for the base model.")
