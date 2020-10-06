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
 - `--num_contextless_addresses=[Number of entries solely consisting of an address, without any context]`
 - `--num_contextless_phones=[Number of entries solely consisting of a phone number, without any context]`
 - `--mask_digits` (to replace all digits with zeros)

The sum of all four probabilities for changes to the case of tokens must not exceed 1.0.
