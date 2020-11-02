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
"""Shared constants and functions."""

LABEL_CONTAINER_NAME = "lucid"

MAIN_LABEL_ADDRESS = "ADDRESS"
MAIN_LABEL_TELEPHONE = "TELEPHONE"
MAIN_LABELS = [MAIN_LABEL_ADDRESS, MAIN_LABEL_TELEPHONE]

# "O" is short for "outside" and a magic value used by seqeval
# Not assigning "O" the index 0 help to find bugs where the label is always set
# to zero.
LABEL_OUTSIDE = "O"
LABEL_BEGIN_TELEPHONE = "B-TELEPHONE"
LABEL_INSIDE_TELEPHONE = "I-TELEPHONE"
LABEL_BEGIN_ADDRESS = "B-ADDRESS"
LABEL_INSIDE_ADDRESS = "I-ADDRESS"
LABELS = [
    LABEL_BEGIN_TELEPHONE, LABEL_INSIDE_TELEPHONE, LABEL_OUTSIDE,
    LABEL_BEGIN_ADDRESS, LABEL_INSIDE_ADDRESS
]
