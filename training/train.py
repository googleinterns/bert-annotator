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
"""Trains the model and saves the final weights."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app, flags
import functools
import tensorflow as tf

from official.nlp.tasks.tagging import TaggingConfig, TaggingTask
from official.nlp.data import tagging_dataloader
from training.utils import labels


def train(module_url, train_data_path, validation_data_path, epochs,
          train_size, save_path, batch_size, optimizer_name, learning_rate):
    train_data_config = tagging_dataloader.TaggingDataConfig(
        input_path=train_data_path,
        seq_length=128,
        global_batch_size=batch_size)
    validation_data_config = tagging_dataloader.TaggingDataConfig(
        input_path=validation_data_path,
        seq_length=128,
        global_batch_size=batch_size,
        is_training=False)
    config = TaggingConfig(hub_module_url=module_url,
                           train_data=train_data_config,
                           validation_data=validation_data_config,
                           class_names=labels)
    task = TaggingTask(config)
    model = task.build_model()
    if optimizer_name == "sgd":
        optimizer = tf.keras.optimizers.SGD(lr=learning_rate)
    else:
        assert optimizer_name == "adam"
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")])
    model.train_step = functools.partial(task.train_step,
                                         model=model,
                                         optimizer=model.optimizer)
    dataset_train = task.build_inputs(config.train_data)
    dataset_validation = task.build_inputs(config.validation_data)

    model.fit(
        dataset_train,
        validation_data=dataset_validation,
        epochs=epochs,
        steps_per_epoch=train_size // batch_size,
    )
    model.save_weights(save_path)


def main(_):
    FLAGS = flags.FLAGS  # pylint: disable=invalid-name

    train(FLAGS.module_url, FLAGS.train_data_path, FLAGS.validation_data_path,
          FLAGS.epochs, FLAGS.train_size, FLAGS.save_path, FLAGS.batch_size,
          FLAGS.optimizer, FLAGS.learning_rate)


if __name__ == "__main__":
    flags.DEFINE_string("module_url", None,
                        "The URL to the pretrained Bert model.")
    flags.mark_flag_as_required("module_url")

    flags.DEFINE_string("train_data_path", None,
                        "The path to the training data in .tfrecord format.")
    flags.mark_flag_as_required("train_data_path")

    flags.DEFINE_string(
        "validation_data_path", None,
        "The path to the validation data in .tfrecord format.")
    flags.mark_flag_as_required("validation_data_path")

    flags.DEFINE_integer(
        "epochs", None, "The maximal number of training epochs. Early stopping"
        " may supercede this setting.")
    flags.mark_flag_as_required("epochs")

    flags.DEFINE_integer("train_size", None,
                         "The number of samples in the training set.")
    flags.mark_flag_as_required("train_size")

    flags.DEFINE_string("save_path", None,
                        "The output path for the final trained model.")
    flags.mark_flag_as_required("save_path")

    flags.DEFINE_integer("batch_size", 64, "The number of samples per batch.")

    flags.DEFINE_string("optimizer", "sgd",
                        "The optimizer. Either 'adam' or 'sgd'")

    flags.DEFINE_float("learning_rate", 0.01, "The learning rate")

    app.run(main)
