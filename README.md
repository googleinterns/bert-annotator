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
 - `--total=[total augmentations]`
 - `--lowercase=[augmentations by lowercasing]`
 - `--addresses_path=[path to address file]`
 - `--addresses=[augmentations by address replacement]`
 - `--phones_path=[path to phone number file]`
 - `--phones=[augmentations by phone number replacement]`
