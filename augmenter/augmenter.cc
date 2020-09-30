//
// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#include "augmenter/augmenter.h"

#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/ascii.h"
#include "augmenter/augmentations.h"
#include "augmenter/random_sampler.h"
#include "augmenter/token_sequence.h"
#include "protocol_buffer/document.pb.h"
#include "protocol_buffer/documents.pb.h"

namespace augmenter {

Augmenter::Augmenter(const bert_annotator::Documents& documents,
                     Augmentations augmentations,
                     RandomSampler* const address_sampler,
                     RandomSampler* const phone_sampler,
                     absl::BitGenRef bitgenref)
    : documents_(documents),
      address_sampler_(address_sampler),
      phone_sampler_(phone_sampler),
      augmentations_(augmentations),
      bitgenref_(bitgenref) {
  // The input uses more detailed address labels. To have a consistent output,
  // all those labels have to be switched to the generall "ADDRESS" label.
  for (auto& document : *documents_.mutable_documents()) {
    if (document.labeled_spans().find("lucid") ==
        document.labeled_spans().end()) {
      continue;
    }
    auto labeled_spans =
        document.mutable_labeled_spans()->at("lucid").mutable_labeled_span();
    for (auto& labeled_span : *labeled_spans) {
      if (kAddressLabels.contains(labeled_span.label())) {
        labeled_span.set_label(
            std::string(Augmenter::kAddressReplacementLabel));
      }
    }
  }
}

Augmenter::Augmenter(const bert_annotator::Documents& documents,
                     Augmentations augmentations,
                     RandomSampler* const address_sampler,
                     RandomSampler* const phone_sampler)
    : Augmenter(documents, augmentations, address_sampler, phone_sampler,
                bitgen_) {}

bool Augmenter::AugmentAddress(
    bert_annotator::Document* const augmented_document) {
  const bool replaced_address = MaybeReplaceLabel(
      static_cast<double>(augmentations_.address) / augmentations_.total,
      address_sampler_, Augmenter::kAddressReplacementLabel,
      augmented_document);
  if (replaced_address) {
    --augmentations_.address;
    return true;
  }
  return false;
}

bool Augmenter::AugmentPhone(
    bert_annotator::Document* const augmented_document) {
  const bool replaced_phone = MaybeReplaceLabel(
      static_cast<double>(augmentations_.phone) / augmentations_.total,
      phone_sampler_, Augmenter::kPhoneReplacementLabel, augmented_document);
  if (replaced_phone) {
    --augmentations_.phone;
    return true;
  }
  return false;
}

bool Augmenter::AugmentLowercase(
    bert_annotator::Document* const augmented_document) {
  const bool perform_lowercasing = absl::Bernoulli(
      bitgenref_,
      static_cast<double>(augmentations_.lowercase) / augmentations_.total);
  if (perform_lowercasing) {
    Lowercase(augmented_document);
    --augmentations_.lowercase;
    return true;
  }
  return false;
}

void Augmenter::Augment() {
  const int original_document_number = documents_.documents_size();
  while (augmentations_.total > 0) {
    const int document_id = absl::Uniform(absl::IntervalClosed, bitgenref_, 0,
                                          original_document_number - 1);
    const bert_annotator::Document& original_document =
        documents_.documents(document_id);
    bert_annotator::Document* augmented_document = documents_.add_documents();
    augmented_document->CopyFrom(original_document);

    bool augmentation_performed = false;
    augmentation_performed |= AugmentAddress(augmented_document);
    augmentation_performed |= AugmentPhone(augmented_document);
    augmentation_performed |= AugmentLowercase(augmented_document);
    augmentation_performed |= AugmentContext(augmented_document);

    // If no action was performed and all remaining augmentations have to
    // perform at least one action, drop this sample. It's identical to the
    // original document. Repeat this augmentation iteration.
    if (!augmentation_performed &&
        augmentations_.total ==
            augmentations_.lowercase + augmentations_.address +
                augmentations_.phone + augmentations_.context_keep_labels +
                augmentations_.context_drop_labels) {
      documents_.mutable_documents()->RemoveLast();
    } else {
      --augmentations_.total;
    }
  }
}

bool Augmenter::AugmentContext(
    bert_annotator::Document* const augmented_document) {
  const bool dropped_context_keeping_labels = MaybeDropContextKeepLabels(
      static_cast<double>(augmentations_.context_keep_labels) /
          augmentations_.total,
      augmented_document);
  if (dropped_context_keeping_labels) {
    --augmentations_.context_keep_labels;
    return true;
  }

  const bool dropped_context_dropping_labels = MaybeDropContextDropLabels(
      static_cast<double>(augmentations_.context_drop_labels) /
          augmentations_.total,
      augmented_document);
  if (dropped_context_dropping_labels) {
    --augmentations_.context_drop_labels;
    return true;
  }

  return false;
}

std::vector<TokenSequence> Augmenter::DropableSequences(
    const bert_annotator::Document& document) {
  if (document.labeled_spans().find("lucid") ==
      document.labeled_spans().end()) {
    return {TokenSequence{.start = 0, .end = document.token_size() - 1}};
  }

  std::vector<TokenSequence> dropable_sequences;
  int start = 0;

  const auto labeled_spans =
      document.labeled_spans().at("lucid").labeled_span();
  for (int i = 0; i < labeled_spans.size(); ++i) {
    const auto& labeled_span = labeled_spans.at(i);
    if (labeled_span.label().compare(
            std::string(Augmenter::kAddressReplacementLabel)) == 0 ||
        labeled_span.label().compare(
            std::string(Augmenter::kPhoneReplacementLabel)) == 0) {
      if (start != labeled_span.token_start()) {
        dropable_sequences.push_back(TokenSequence{
            .start = start, .end = labeled_span.token_start() - 1});
      }
      start = labeled_span.token_end() + 1;
    }
  }
  if (start != document.token_size()) {
    dropable_sequences.push_back(
        TokenSequence{.start = start, .end = document.token_size() - 1});
  }

  return dropable_sequences;
}

std::vector<TokenSequence> Augmenter::LabeledSequences(
    const bert_annotator::Document& document) {
  if (document.labeled_spans().find("lucid") ==
      document.labeled_spans().end()) {
    return {TokenSequence{.start = 0, .end = document.token_size() - 1}};
  }

  std::vector<TokenSequence> labeled_sequences;

  const auto labeled_spans =
      document.labeled_spans().at("lucid").labeled_span();
  for (int i = 0; i < labeled_spans.size(); ++i) {
    const auto& labeled_span = labeled_spans.at(i);
    if (labeled_span.label().compare(
            std::string(Augmenter::kAddressReplacementLabel)) == 0 ||
        labeled_span.label().compare(
            std::string(Augmenter::kPhoneReplacementLabel)) == 0) {
      labeled_sequences.push_back(
          TokenSequence{.start = labeled_span.token_start(),
                        .end = labeled_span.token_end()});
    }
  }
  return labeled_sequences;
}

bool Augmenter::MaybeDropContextKeepLabels(
    const double probability,
    bert_annotator::Document* const augmented_document) {
  const bool do_drop_context = absl::Bernoulli(bitgenref_, probability);
  if (!do_drop_context) {
    return false;
  }

  bool dropped_context = false;
  std::vector<TokenSequence> dropable_sequences =
      DropableSequences(*augmented_document);
  // If there are not labels, we will get a single dropable sequence containing
  // the whole sentence. To drop from both its start and end, we duplicate it
  // here.
  bool no_labels = false;
  if (dropable_sequences.size() == 1 && dropable_sequences[0].start == 0 &&
      dropable_sequences[0].end == augmented_document->token_size() - 1) {
    no_labels = true;
    dropable_sequences.push_back(dropable_sequences[0]);
  }

  // Tokens will be dropped, so iterating backwards avoids the need to
  // update the indices in dropable_sequences.
  for (int i = dropable_sequences.size() - 1; i >= 0; --i) {
    const auto& dropable_sequence = dropable_sequences.at(i);
    const bool do_drop = absl::Bernoulli(bitgenref_, 0.5);
    if (!do_drop) {
      continue;
    }

    // At least one token should be dropped, so the sequence needs to be at
    // least two tokens long.
    if (dropable_sequence.end == dropable_sequence.start) {
      continue;
    }

    dropped_context = true;

    int drop_tokens_start;
    int drop_tokens_end;
    if (i == static_cast<int>(dropable_sequences.size()) - 1 &&
        dropable_sequence.end == augmented_document->token_size() -
                                     1) {  // For context after the last label,
                                           // drop a postfix of the sentence.
      drop_tokens_start =
          absl::Uniform(absl::IntervalClosed, bitgenref_,
                        dropable_sequence.start + 1, dropable_sequence.end);
      drop_tokens_end = dropable_sequence.end;
      // If no labels exist, the sequences was duplicated, so the index needs to
      // be updated.
      if (no_labels) {
        dropable_sequences[0].end = drop_tokens_start;
      }
    } else if (dropable_sequence.start ==
               0) {  // For context before the first label, drop a prefix of the
                     // sentence.
      drop_tokens_start = 0;
      drop_tokens_end = absl::Uniform(absl::IntervalClosed, bitgenref_, 0,
                                      dropable_sequence.end - 1);
    } else {  // For context between two labels, drop a subsequence.
      if (dropable_sequence.end - dropable_sequence.start ==
          1) {  // At least the first and the last tokens must not be dropped.
        continue;
      }

      drop_tokens_start =
          absl::Uniform(absl::IntervalClosed, bitgenref_,
                        dropable_sequence.start + 1, dropable_sequence.end - 1);
      drop_tokens_end =
          absl::Uniform(absl::IntervalClosed, bitgenref_, drop_tokens_start,
                        dropable_sequence.end - 1);
    }

    const TokenSequence boundaries =
        TokenSequence{.start = drop_tokens_start, .end = drop_tokens_end};
    const int removed_characters = DropText(boundaries, augmented_document);
    DropTokens(boundaries, augmented_document);
    ShiftTokenBoundaries(boundaries.start, -removed_characters,
                         augmented_document);
    UpdateLabeledSpansForDroppedTokens(boundaries, augmented_document);
  }

  return dropped_context;
}

bool Augmenter::MaybeDropContextDropLabels(
    const double probability,
    bert_annotator::Document* const augmented_document) {
  const int token_count = augmented_document->token_size();
  std::vector<TokenSequence> labeled_sequences =
      LabeledSequences(*augmented_document);
  // MaybeDropContextKeepLabels already implements dropping from sentences
  // without any labels.
  if (labeled_sequences.size() == 0) {
    return MaybeDropContextKeepLabels(probability, augmented_document);
  }

  const bool do_drop_context = absl::Bernoulli(bitgenref_, probability);
  if (!do_drop_context || token_count == 1) {
    return false;
  }

  bool dropped_context = false;
  const int label_id =
      absl::Uniform(bitgenref_, 0, static_cast<int>(labeled_sequences.size()));
  TokenSequence labeled_sequence = labeled_sequences[label_id];
  if (labeled_sequence.end < token_count - 2) {
    TokenSequence drop_sequence{.start = 0, .end = token_count - 1};
    const bool drop_right = absl::Bernoulli(bitgenref_, 0.5);
    if (drop_right) {
      drop_sequence.start =
          absl::Uniform(absl::IntervalClosed, bitgenref_,
                        labeled_sequence.end + 2, token_count - 1);
      const int removed_characters =
          DropText(drop_sequence, augmented_document);
      DropTokens(drop_sequence, augmented_document);
      ShiftTokenBoundaries(drop_sequence.start, -removed_characters,
                           augmented_document);
      UpdateLabeledSpansForDroppedTokens(drop_sequence, augmented_document);
      dropped_context = true;
    }
  }
  if (labeled_sequence.start > 1) {
    TokenSequence drop_sequence{.start = 0, .end = 0};
    const bool drop_left = absl::Bernoulli(bitgenref_, 0.5);
    if (drop_left) {
      drop_sequence.end = absl::Uniform(absl::IntervalClosed, bitgenref_, 0,
                                        labeled_sequence.start - 2);
      const int removed_characters =
          DropText(drop_sequence, augmented_document);
      DropTokens(drop_sequence, augmented_document);
      ShiftTokenBoundaries(drop_sequence.start, -removed_characters,
                           augmented_document);
      UpdateLabeledSpansForDroppedTokens(drop_sequence, augmented_document);
      dropped_context = true;
    }
  }

  return dropped_context;
}

bool Augmenter::MaybeReplaceLabel(const double probability,
                                  RandomSampler* const sampler,
                                  const absl::string_view label,
                                  bert_annotator::Document* const document) {
  const std::vector<TokenSequence>& boundary_list =
      LabelBoundaryList(*document, label);
  const bool do_replace = absl::Bernoulli(bitgenref_, probability);
  if (do_replace && !boundary_list.empty()) {
    const int boundary_index =
        absl::Uniform(absl::IntervalClosed, bitgenref_, static_cast<size_t>(0),
                      boundary_list.size() - 1);
    const TokenSequence boundaries = boundary_list[boundary_index];
    const std::string replacement = sampler->Sample();
    ReplaceLabeledTokens(boundaries, replacement, label, document);
    return true;
  }
  return false;
}

const int Augmenter::ReplaceText(
    const TokenSequence& boundaries, const std::string& replacement,
    bert_annotator::Document* const document) const {
  const int string_start = document->token(boundaries.start).start();
  const int string_end = document->token(boundaries.end).end();
  std::string new_text;
  new_text.append(document->mutable_text()->begin(),
                  document->mutable_text()->begin() + string_start);
  new_text.append(replacement);
  if (static_cast<int>(document->text().size()) > string_end) {
    new_text.append(document->mutable_text()->begin() + string_end + 1,
                    document->mutable_text()->end());
  }
  document->set_text(new_text);

  return replacement.size() - (string_end - string_start + 1);
}

const int Augmenter::DropText(const TokenSequence& boundaries,
                              bert_annotator::Document* const document) const {
  int text_start;
  int text_end;
  if (boundaries.start > 0) {
    text_start = document->token(boundaries.start - 1).end() + 1;
    text_end = document->token(boundaries.end).end();
  } else {
    text_start = 0;
    if (boundaries.end < document->token_size() - 1) {
      text_end = document->token(boundaries.end + 1).start() - 1;
    } else {
      text_end = document->text().size() - 1;
    }
  }

  std::string new_text;
  new_text.append(document->mutable_text()->begin(),
                  document->mutable_text()->begin() + text_start);
  if (static_cast<int>(document->text().size()) > text_end) {
    new_text.append(document->mutable_text()->begin() + text_end + 1,
                    document->mutable_text()->end());
  }
  document->set_text(new_text);

  return text_end - text_start + 1;
}

void Augmenter::ReplaceToken(const int token_id, const std::string& replacement,
                             bert_annotator::Document* const document) const {
  auto token = document->mutable_token(token_id);
  token->set_word(replacement);
  token->set_end(token->start() + replacement.size() - 1);
}

void Augmenter::DropTokens(const TokenSequence boundaries,
                           bert_annotator::Document* const document) const {
  document->mutable_token()->erase(
      document->mutable_token()->begin() + boundaries.start,
      document->mutable_token()->begin() + boundaries.end +
          1);  // end is inclusive.
}

void Augmenter::ShiftTokenBoundaries(
    const int first_token, const int shift,
    bert_annotator::Document* const document) const {
  for (int i = first_token; i < document->token_size(); ++i) {
    auto token = document->mutable_token(i);
    token->set_start(token->start() + shift);
    token->set_end(token->end() + shift);
  }
}

void Augmenter::ReplaceLabeledSpan(
    const int token_id, const absl::string_view replacement_label,
    bert_annotator::Document* const document) const {
  const auto& labeled_spans =
      document->mutable_labeled_spans()->at("lucid").mutable_labeled_span();
  for (auto& labeled_span : *labeled_spans) {
    if (labeled_span.token_start() == token_id) {
      labeled_span.set_label(std::string(replacement_label));
      labeled_span.set_token_end(labeled_span.token_start());
    }
  }
}

void Augmenter::UpdateLabeledSpansForDroppedTokens(
    const TokenSequence& removed_tokens,
    bert_annotator::Document* const document) const {
  if (document->labeled_spans().find("lucid") ==
      document->labeled_spans().end()) {
    return;
  }

  const auto& labeled_spans =
      document->mutable_labeled_spans()->at("lucid").mutable_labeled_span();
  for (int i = labeled_spans->size() - 1; i >= 0; --i) {
    auto labeled_span = labeled_spans->Mutable(i);
    if (labeled_span->token_end() <
        removed_tokens.start) {  // Label not affected by removed tokens.
      continue;
    } else if (labeled_span->token_start() <=
               removed_tokens.end) {  // Complete label removed.
      labeled_spans->erase(labeled_spans->begin() + i);
    } else {  // Label remains but needs to be shifted.
      labeled_span->set_token_start(
          labeled_span->token_start() -
          (removed_tokens.end - removed_tokens.start + 1));
      labeled_span->set_token_end(
          labeled_span->token_end() -
          (removed_tokens.end - removed_tokens.start + 1));
    }
  }
}

void Augmenter::ReplaceLabeledTokens(
    const TokenSequence& boundaries, const std::string& replacement,
    const absl::string_view replacement_label,
    bert_annotator::Document* const document) const {
  const int text_shift = ReplaceText(boundaries, replacement, document);
  ReplaceToken(boundaries.start, replacement, document);
  DropTokens(
      TokenSequence{.start = boundaries.start + 1, .end = boundaries.end},
      document);
  ShiftTokenBoundaries(boundaries.start + 1, text_shift, document);
  ReplaceLabeledSpan(boundaries.start, replacement_label, document);
  UpdateLabeledSpansForDroppedTokens(
      TokenSequence{.start = boundaries.start + 1, .end = boundaries.end},
      document);
}

const std::vector<TokenSequence> Augmenter::LabelBoundaryList(
    const bert_annotator::Document& document,
    const absl::string_view label) const {
  if (document.labeled_spans().find("lucid") ==
      document.labeled_spans().end()) {
    return {};
  }
  const auto labeled_spans =
      document.labeled_spans().at("lucid").labeled_span();

  // First, select only spans labeled as one of the given labels. Then, join
  // subsequent spans.
  std::vector<TokenSequence> boundary_list = {};
  for (int i = 0; i < labeled_spans.size(); ++i) {
    const auto labeled_span = labeled_spans[i];
    if (labeled_span.label().compare(std::string(label)) == 0) {
      boundary_list.push_back(TokenSequence{.start = labeled_span.token_start(),
                                            .end = labeled_span.token_end()});
    }
  }
  for (int i = boundary_list.size() - 2; i >= 0; --i) {
    if (boundary_list[i].end + 1 == boundary_list[i + 1].start) {
      boundary_list[i].end = boundary_list[i + 1].end;
      boundary_list.erase(boundary_list.begin() + i + 1);
    }
  }

  return boundary_list;
}

void Augmenter::Lowercase(
    bert_annotator::Document* const augmented_document) const {
  std::string* const text = augmented_document->mutable_text();
  std::string new_text;
  int text_index = 0;
  for (int j = 0; j < augmented_document->token_size(); ++j) {
    bert_annotator::Token* const token = augmented_document->mutable_token(j);

    // Adds the string in between two tokens as it is.
    const int token_start = token->start();
    const int token_end = token->end();
    if (text_index < token_start) {
      new_text.append(text->begin() + text_index, text->begin() + token_start);
    }

    // Transforms the token to lowercase.
    std::string* const word = token->mutable_word();
    absl::AsciiStrToLower(word);
    new_text.append(*word);
    text_index = token_end + 1;
  }
  new_text.append(text->begin() + text_index, text->end());
  augmented_document->set_text(new_text);
}

const bert_annotator::Documents Augmenter::documents() const {
  return documents_;
}

const absl::flat_hash_set<absl::string_view>& Augmenter::kAddressLabels =
    *new absl::flat_hash_set<absl::string_view>(
        {"LOCALITY", "COUNTRY", "ADMINISTRATIVE_AREA", "THOROUGHFARE",
         "THOROUGHFARE_NUMBER", "PREMISE", "POSTAL_CODE", "PREMISE_LEVEL"});

constexpr absl::string_view Augmenter::kAddressReplacementLabel;

const absl::flat_hash_set<absl::string_view>& Augmenter::kPhoneLabels = {
    "TELEPHONE"};

constexpr absl::string_view Augmenter::kPhoneReplacementLabel;

}  // namespace augmenter
