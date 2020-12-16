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
from official.nlp.tasks.tagging import TaggingConfig
from official.nlp.data import tagging_dataloader
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
from training.configurable_training_tagging_task import (
    ConfigurableTrainingTaggingTask)
from training.utils import LABELS, ADDITIONAL_LABELS

flags.DEFINE_string("module_url", None,
                    "The URL to the pretrained Bert model.")
flags.DEFINE_string("train_data_path", None,
                    "The path to the training data in .tfrecord format.")
flags.DEFINE_string("validation_data_path", None,
                    "The path to the validation data in .tfrecord format.")
flags.DEFINE_integer("epochs", None, "The maximal number of training epochs.")
flags.DEFINE_integer("train_size", None,
                     "The number of samples in the training set.")
flags.DEFINE_string("save_path", None,
                    "The output path for the final trained model.")
flags.DEFINE_integer("batch_size", 64, "The number of samples per batch.")
flags.DEFINE_enum("optimizer", "sgd", ["sgd", "adam"], "The optimizer.")
flags.DEFINE_float("learning_rate", 0.01, "The learning rate.")
flags.DEFINE_float(
    "plateau_lr_reduction", 1.0,
    "The learning rate is reduced by this factor once a plateau (measured on"
    " the validation loss) is reached)")
flags.DEFINE_integer(
    "plateau_patience", 3,
    "How many epochs to wait on a plateau before the learning rate is reduced."
)
flags.DEFINE_boolean(
    "train_with_additional_labels", False,
    "If set, the flags other than address/phone are used, too.")
flags.DEFINE_boolean("train_last_layer_only", False,
                     "If set, only the last layer is trainable.")
flags.DEFINE_string(
    "tpu_address", None,
    "The internal address of the TPU node, including 'grpc://'. If not set, no"
    " tpu is used.")

FLAGS = flags.FLAGS


def train(module_url, train_data_path, validation_data_path, epochs,
          train_size, save_path, batch_size, optimizer_name, learning_rate,
          train_last_layer_only, plateau_lr_reduction, plateau_patience,
          tpu_address):
    if tpu_address is not None:
        if plateau_lr_reduction != 1.0:
            raise NotImplementedError(
                "Learning rate reduction cannot be used on TPUs, because the"
                " validation set cannot be evaluated.")
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
            tpu=tpu_address)
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)
    else:
        if plateau_lr_reduction != 1.0 and validation_data_path is None:
            raise ValueError(
                "In order to reduce the learning rate on plateaus, a validation"
                " set must be specified.")
        strategy = tf.distribute.get_strategy()

    with strategy.scope():
        train_data_config = tagging_dataloader.TaggingDataConfig(
            input_path=train_data_path,
            seq_length=128,
            global_batch_size=batch_size)
        if validation_data_path is not None:
            validation_data_config = tagging_dataloader.TaggingDataConfig(
                input_path=validation_data_path,
                seq_length=128,
                global_batch_size=batch_size,
                is_training=False)
        else:
            validation_data_config = None

        label_list = LABELS
        if FLAGS.train_with_additional_labels:
            label_list = LABELS + ADDITIONAL_LABELS
        config = TaggingConfig(hub_module_url=module_url,
                               train_data=train_data_config,
                               validation_data=validation_data_config,
                               class_names=label_list)
        task = ConfigurableTrainingTaggingTask(config)
        model = task.build_model(train_last_layer_only)
        if optimizer_name == "sgd":
            optimizer = tf.keras.optimizers.SGD(lr=learning_rate)
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        iterations_per_epoch = train_size // batch_size
        model.compile(
            optimizer=optimizer,
            metrics=[
                tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")
            ],
            steps_per_execution=iterations_per_epoch)
        model.train_step = functools.partial(task.train_step,
                                             model=model,
                                             optimizer=model.optimizer)
        dataset_train = task.build_inputs(config.train_data)

        checkpoint = ModelCheckpoint(save_path + "/model_{epoch:02d}",
                                     verbose=1,
                                     save_best_only=False,
                                     save_weights_only=True,
                                     period=1)
        callbacks = [checkpoint]

        additional_fit_parameters = {}
        if plateau_lr_reduction != 1.0:
            dataset_validation = task.build_inputs(config.validation_data)
            reduce_lr = ReduceLROnPlateau(monitor="val_loss",
                                          factor=plateau_lr_reduction,
                                          patience=plateau_patience,
                                          verbose=1)
            callbacks.append(reduce_lr)
            additional_fit_parameters["validation_data"] = dataset_validation
            model.test_step = functools.partial(task.validation_step,
                                                model=model)

        model.fit(dataset_train,
                  epochs=epochs,
                  steps_per_epoch=iterations_per_epoch,
                  callbacks=callbacks,
                  **additional_fit_parameters)


def main(_):
    train(FLAGS.module_url, FLAGS.train_data_path, FLAGS.validation_data_path,
          FLAGS.epochs, FLAGS.train_size, FLAGS.save_path, FLAGS.batch_size,
          FLAGS.optimizer, FLAGS.learning_rate, FLAGS.train_last_layer_only,
          FLAGS.plateau_lr_reduction, FLAGS.plateau_patience,
          FLAGS.tpu_address)


if __name__ == "__main__":
    flags.mark_flag_as_required("module_url")
    flags.mark_flag_as_required("train_data_path")
    flags.mark_flag_as_required("epochs")
    flags.mark_flag_as_required("train_size")
    flags.mark_flag_as_required("save_path")

    app.run(main)
