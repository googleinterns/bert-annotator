#!/usr/bin/env python3

#
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Modified version of the TaggingTask to allow only training the last layer."""

from official.nlp.configs import encoders
from official.nlp.modeling import models
from official.nlp.tasks import utils
from official.nlp.tasks.tagging import TaggingTask
import tensorflow as tf


class ConfigurableTrainingTaggingTask(TaggingTask):
    """TaggingTask with the option to only train the last layer."""
    def build_model(self, train_last_layer_only=False):
        """Modified version of official.nlp.tasks.tagging.build_model

        Allows to freeze the underlying bert encoder, such that only the dense
        layer is trained.
        """
        if self.task_config.hub_module_url and self.task_config.init_checkpoint:
            raise ValueError("At most one of `hub_module_url` and "
                             "`init_checkpoint` can be specified.")
        if self.task_config.hub_module_url:
            encoder_network = utils.get_encoder_from_hub(
                self.task_config.hub_module_url)
        else:
            encoder_network = encoders.build_encoder(
                self.task_config.model.encoder)
        encoder_network.trainable = not train_last_layer_only

        return models.BertTokenClassifier(
            network=encoder_network,
            num_classes=len(self.task_config.class_names),
            initializer=tf.keras.initializers.TruncatedNormal(
                stddev=self.task_config.model.head_initializer_range),
            dropout_rate=self.task_config.model.head_dropout,
            output="logits")
