# Bert-Annotator

A possible improvement for entity selection and classification

## Augmenter

Augments available corpora by replacing entities or performing string
manipulations

### Setup

First, clone this repository using 

```sh
git clone git@github.com:googleinterns/bert-annotator.git
```

Next, create the following files:
 - `protocol_buffer/document.proto` containing the protocol buffer definitions
 - `data/input/raw/[corpus].textproto` one or multiple corpus files that should be
   augmented

Now, build the augmenter using

```sh
bazel build //augmenter:main
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
 - `--corpora="[corpora]"`
 - `--num_total=[total augmentations]`
 - `--num_lowercasings_complete_token=[augmentations by lowercasing complete tokens]`
 - `--probability_per_lowercasing_complete_token=[given that some tokens are completely lowercased, how likely is each token to be affected? Defaults to 0.5]`
 - `--num_lowercasings_first_letter=[augmentations by lowercasing the first letter of tokens]`
 - `--probability_per_lowercasing_first_letter=[given that the first letter of some tokens will be lowercased, how likely is each token to be affected? Defaults to 0.5]`
 - `--num_uppercasings_complete_token=[augmentations by uppercasing complete tokens]`
 - `--probability_per_uppercasing_complete_token=[given that some tokens are completely uppercased, how likely is each token to be affected? Defaults to 0.5]`
 - `--num_uppercasings_first_letter=[augmentations by uppercasing the first letter of tokens]`
 - `--probability_per_uppercasing_first_letter=[given that the first letter of some tokens will be uppercased, how likely is each token to be affected? Defaults to 0.5]`
 - `--addresses_path=[path to address file]`
 - `--num_address_replacements=[augmentations by address replacement]`
 - `--phones_path=[path to phone number file]`
 - `--num_phone_replacements=[augmentations by phone number replacement]`
 - `--num_context_drops_between_labels=[augmentations by dropping context in between labels. Keeps at least the token directly to the left and right of each label.]`
 - `--num_context_drops_outside_one_label=[augmentations by selecting a label and dropping context to its left and right. May drop other labels]`
 - `--probability_per_drop=[given that context from a sentence will be dropped, how likely is each sequence to be dropped? Defaults to 0.5]`
 - `--num_contextless_addresses=[entries solely consisting of an address, without any context]`
 - `--num_contextless_phones=[entries solely consisting of a phone number, without any context]`
 - `--mask_digits` (to replace all digits with zeros)
