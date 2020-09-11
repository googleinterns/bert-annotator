# Bert-Annotator

A possible improvement for entity selection and classification

## Augmenter

Augments available corpora by replacing entities or performing string
manipulations

### Setup

First, clone this repository using 

```
git clone git@github.com:googleinterns/bert-annotator.git
```

Next, create the following files:
 - `protocol_buffer/document.proto` containing the protocol buffer definitions
 - `input_data/[corpus].textproto` one or multiple corpus files that should be
   augmented

Now, build the augmenter using

```
bazel build //augmenter::main
```

Because the exported `textproto` files don't have the correct format,
preprocess them first:

```
bash input_data/proprocess.sh [corpus]
```

### Execute

The augmenter can be run using

```
./bazel-bin/augmenter/main [corpus]
```
