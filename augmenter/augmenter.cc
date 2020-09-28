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

void Augmenter::Augment() {
  const int original_document_number = documents_.documents_size();
  while (augmentations_.total > 0) {
    const int document_id =
        absl::Uniform(bitgenref_, 0, original_document_number - 1);
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
        augmentations_.total == augmentations_.lowercase +
                                    augmentations_.address +
                                    augmentations_.phone) {
      documents_.mutable_documents()->RemoveLast();
    } else {
      --augmentations_.total;
    }
  }
}

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

bool Augmenter::AugmentContext(
    bert_annotator::Document* const augmented_document) {
  const bool dropped_context = MaybeDropContext(
      static_cast<double>(augmentations_.context) / augmentations_.total,
      augmented_document);
  if (dropped_context) {
    --augmentations_.context;
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
    auto labeled_span = labeled_spans.at(i);
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

bool Augmenter::MaybeDropContext(
    const double probability,
    bert_annotator::Document* const augmented_document) {
  const bool do_drop_context = absl::Bernoulli(bitgenref_, probability);
  if (!do_drop_context) {
    return false;
  }

  bool dropped_context = false;
  std::vector<TokenSequence> dropable_sequences =
      DropableSequences(*augmented_document);
  // Tokens will be dropped, so iterating backwards avoids the need to update
  // the indices in dropable_sequences.
  for (int i = dropable_sequences.size() - 1; i >= 0; --i) {
    auto dropable_sequence = dropable_sequences.at(i);
    std::cerr << "A (" << std::to_string(dropable_sequence.start) << " - "
              << std::to_string(dropable_sequence.end) << ")" << std::endl;
    bool do_drop = absl::Bernoulli(bitgenref_, 0.5);
    if (!do_drop) {
      continue;
    }
    std::cerr << "B" << std::endl;

    // At least one token should be dropped, so the sequence needs to be at
    // least two tokens long.
    if (dropable_sequence.end == dropable_sequence.start) {
      continue;
    }
    std::cerr << "C" << std::endl;

    dropped_context = true;

    // Drop some words from the left side, drop some from the right side.
    // This ensures that at least one token remains, while making the specific
    // choice which one remains random.
    int tokens_to_drop_left = absl::Uniform(
        bitgenref_, 0, dropable_sequence.end - dropable_sequence.start);

    if (tokens_to_drop_left > 0) {
      TokenSequence boundaries_left = TokenSequence{
          .start = dropable_sequence.start,
          .end = dropable_sequence.start + tokens_to_drop_left - 1};
      const int removed_characters_left =
          DropText(boundaries_left, augmented_document);
      DropTokens(augmented_document, boundaries_left);
      ShiftTokenBoundaries(boundaries_left.start, -removed_characters_left,
                           augmented_document);
      UpdateLabeledSpansForDroppedTokens(boundaries_left, augmented_document);
      dropable_sequence.end -= tokens_to_drop_left;
    }

    if (dropable_sequence.end == dropable_sequence.start) {
      continue;
    }
    std::cerr << "D" << std::endl;

    int tokens_to_drop_right = absl::Uniform(
        bitgenref_, 0, dropable_sequence.end - dropable_sequence.start);
    if (tokens_to_drop_right > 0) {
      TokenSequence boundaries_right = TokenSequence{
          .start = dropable_sequence.end - tokens_to_drop_right + 1,
          .end = dropable_sequence.end};
      const int removed_characters_right =
          DropText(boundaries_right, augmented_document);
      DropTokens(augmented_document, boundaries_right);
      ShiftTokenBoundaries(boundaries_right.start, -removed_characters_right,
                           augmented_document);
      UpdateLabeledSpansForDroppedTokens(boundaries_right, augmented_document);
    }
    std::cerr << "E" << std::endl;
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
    const int boundary_index = absl::Uniform(bitgenref_, static_cast<size_t>(0),
                                             boundary_list.size() - 1);
    const TokenSequence boundaries = boundary_list[boundary_index];
    const std::string replacement = sampler->Sample();
    ReplaceLabeledTokens(boundaries, replacement, label, document);
    return true;
  }
  return false;
}

void Augmenter::Lowercase(
    bert_annotator::Document* const augmented_document) const {
  std::string* const text = augmented_document->mutable_text();
  std::string new_text;
  int text_index = 0;
  for (int j = 0; j < augmented_document->token_size(); ++j) {
    bert_annotator::Token* const token = augmented_document->mutable_token(j);

    // Adds the string inbetween two tokens as it is.
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

void Augmenter::DropTokens(bert_annotator::Document* const document,
                           TokenSequence boundaries) const {
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
  auto labeled_spans =
      document->mutable_labeled_spans()->at("lucid").mutable_labeled_span();
  for (auto labeled_span : *labeled_spans) {
    if (labeled_span.token_start() == token_id) {
      labeled_span.set_label(std::string(replacement_label));
      labeled_span.set_token_end(labeled_span.token_start());
    }
  }
}

void Augmenter::UpdateLabeledSpansForDroppedTokens(
    const TokenSequence& boundaries,
    bert_annotator::Document* const document) const {
  if (document->labeled_spans().find("lucid") ==
      document->labeled_spans().end()) {
    return;
  }

  auto labeled_spans =
      document->mutable_labeled_spans()->at("lucid").mutable_labeled_span();
  for (int i = labeled_spans->size() - 1; i >= 0; --i) {
    auto labeled_span = labeled_spans->Mutable(i);
    if (labeled_span->token_end() <
        boundaries.start) {  // Label not affected by removed tokens.
      continue;
    } else if (labeled_span->token_start() < boundaries.start &&
               labeled_span->token_end() <=
                   boundaries.end) {  // Beginning of label remains.
      labeled_span->set_token_end(boundaries.start - 1);
    } else if (labeled_span->token_start() < boundaries.start &&
               labeled_span->token_end() >
                   boundaries.end) {  // Beginning and end remain, middle was
                                      // removed.
      labeled_span->set_token_end(labeled_span->token_end() -
                                  (boundaries.end - boundaries.start + 1));
    } else if (labeled_span->token_start() >= boundaries.start &&
               labeled_span->token_end() <=
                   boundaries.end) {  // Complete label removed.
      labeled_spans->erase(labeled_spans->begin() + i);
    } else if (labeled_span->token_start() <= boundaries.end &&
               labeled_span->token_end() >
                   boundaries.end) {  // End of label remains.
      labeled_span->set_token_end(boundaries.start - 1);
    } else {  // Label remains but needs to be shifted.
      labeled_span->set_token_start(labeled_span->token_start() -
                                    (boundaries.end - boundaries.start + 1));
      labeled_span->set_token_end(labeled_span->token_end() -
                                  (boundaries.end - boundaries.start + 1));
    }
  }
}

void Augmenter::ReplaceLabeledTokens(
    const TokenSequence& boundaries, const std::string& replacement,
    const absl::string_view replacement_label,
    bert_annotator::Document* const document) const {
  const int text_shift = ReplaceText(boundaries, replacement, document);
  ReplaceToken(boundaries.start, replacement, document);
  DropTokens(document, TokenSequence{.start = boundaries.start + 1,
                                     .end = boundaries.end});
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
