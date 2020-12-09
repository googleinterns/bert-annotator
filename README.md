# Bert-Annotator

A possible improvement for entity selection and classification

## General Setup

First, clone this repository using 

```sh
git clone git@github.com:googleinterns/bert-annotator.git
```

Next, create the following files:
 - `protocol_buffer/document.proto` containing the protocol buffer definitions
 - `data/input/raw/[corpus].textproto` one or multiple corpus files that should
   be augmented

## Augmenter

Augments available corpora by replacing entities or performing string
manipulations

### Setup

Build the augmenter using

```sh
bazel build -c opt //augmenter:main
```

Because the exported `textproto` files don't have the correct format,
preprocess them first:

```sh
bash data/input/preprocess.sh [corpus]
```

### Execute

The augmenter can be run using

```sh
./bazel-bin/augmenter/main [FLAGS]
```

Valid flags are:
 - `--corpora="[Corpora]"`
 - `--addresses_path=[Path to list of alternative addresses]`
 - `--phones_path=[Path to list of alternative phone numbers]`
 - `--num_total=[Number of total augmentations]`
 - `--prob_lowercasing_complete_token=[Probability of lowercasing a complete token]`
 - `--prob_lowercasing_first_letter=[Probability of lowercasing the first letter of a token]`
 - `--prob_uppercasing_complete_token=[Probability of uppercasing a complete token]`
 - `--prob_uppercasing_first_letter=[Probability of uppercasing the first letter of a token]`
 - `--prob_address_replacement=[Probability of replacing an address]`
 - `--prob_phone_replacement=[Probability of replacing a phone number]`
 - `--prob_context_drop_between_labels=[Probability of dropping context in between labels. Keeps at least the token directly to the left and right of each label]`
 - `--prob_context_drop_outside_one_label=[Probability of selecting a label and dropping context to its left and right. May drop other labels]`
 - `--prob_punctuation_change_between_tokens=[Probability of changing the punctuation between tokens to be one of {", ", ". ", "; ", " - "}]`
 - `--prob_punctuation_change_at_sentence_end=[Probability of changing the punctuation at the sentence end to be one of {"?", "!", ".", ":", ";", " - "}]`
 - `--prob_sentence_concatenation=[Probability of concatenating sentences]`
 - `--num_contextless_addresses=[Number of entries solely consisting of an address, without any context]`
 - `--num_contextless_phones=[Number of entries solely consisting of a phone number, without any context]`
 - `--mask_digits` (to replace all digits with zeros)
 - `--save_as_text` (to save the augmented data in text format, useful for debugging)

The sum of all four probabilities for changes to the case of tokens must not exceed 1.0.


## Training

Training for address and phone number detection

### Setup

Clone `https://github.com/tensorflow/models`, install `tf-nightly` and resolve possible issues with CUDA/CuDNN (install most recent version).

Build the scripts using

```sh
bazel build -c opt --noexperimental_python_import_all_repositories //training:...
```

#### Execute

Transform the (augmented or original) data into TFRecord files

```sh
./bazel-bin/training/convert_data \
 --module_url=[MODULE URL] \
 --train_data_input_path=data/output/train.binproto \
 --dev_data_input_path=data/permanent_output/dev.binproto \
 --test_data_input_paths=data/output/test.binproto \
 --test_data_input_paths=data/input/raw/test_phone.lftxt \
 --test_data_input_paths=data/input/raw/test_address.lftxt \
 --train_data_output_path=data/output/train.tfrecord \
 --dev_data_output_path=data/output/dev.tfrecord \
 --test_data_output_paths=data/output/test_lucidsky.tfrecord \
 --test_data_output_paths=data/output/test_phone.tfrecord \
 --test_data_output_paths=data/output/test_address.tfrecord \
 --meta_data_file_path=data/output/meta_data_file.test
```

For the uncased BERT-base model, use `https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2`, for cased use `https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/2`.


Finally, start the training.

```sh
./bazel-bin/training/train \
 --module_url=[MODULE URL] \
 --train_data_path=data/output/train.tfrecord \
 --validation_data_path=data/output/dev.tfrecord \
 --epochs=4 \
 --train_size=55761 \
 --save_path=data/output/trained_model \
 --batch_size=64
```

And evaluate:

```sh
./bazel-bin/training/evaluate \
 --module_url=[MODULE URL] \
 --model_path=data/output/trained_model \
 --input_paths=data/output/test_lucidsky.tfrecord \
 --raw_paths=data/output/test.binproto \
 --input_paths=data/output/test_address.tfrecord \
 --raw_paths=data/input/raw/test_address.lftxt \
 --input_paths=data/output/test_phone.tfrecord \
 --raw_paths=data/input/raw/test_phone.lftxt \
 --visualisation_folder=visualisation/
```

You can also evaluate pre-generated labels if they were stored as linkfragments.
As the `input_paths`, pass a path to a directory containing one or more
linkfragment files.

```sh
./bazel-bin/training/evaluate \
 --module_url=[MODULE URL] \
 --input_paths=directory_with_linkfragments/ \
 --raw_paths=data/input/raw/test_phone.lftxt \
 --visualisation_folder=visualisation/
```

#### TPU support
To use TPUs for training and evaluation:
 - Ensure that the VM has cloud API access to the compute enginge and storage.
 - Use the flag `--tpu_ip` to define the internal IP address of the TPU node.
 - Set the environment variable `TFHUB_CACHE_DIR` to a directory in a storage
   bucket.
 - Only pass paths to storage buckets for datasets and checkpoints.
