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
#include "absl/strings/str_replace.h"
#include "augmenter/augmentations.h"
#include "augmenter/random_sampler.h"
#include "augmenter/token_range.h"
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
  for (bert_annotator::Document& document : *documents_.mutable_documents()) {
    if (augmentations.mask_digits) {
      MaskDigits(&document);
    }

    // Some tokens only contain separator characters like "," or ".". Keeping
    // track of those complicates the identification of longer labels, because
    // those separators may split longer labels into multiple short ones. By
    // ignoring the separators, this can be avoided. It also avoids that
    // context dropping *only* drops punctuation.
    for (int i = document.token_size() - 1; i >= 0; --i) {
      const bert_annotator::Token& token = document.token(i);
      // TODO(brix): depends on the installed C locale, may need to be changed
      // for non-english languages.
      if (absl::c_none_of(token.word(), [](unsigned char c) {
            return std::isdigit(c) || std::isalpha(c);
          })) {
        const TokenRange removed_tokens = TokenRange{.start = i, .end = i};
        DropTokens(removed_tokens, &document);
        UpdateLabeledSpansForDroppedTokens(removed_tokens, &document);
      }
    }

    // The input uses more detailed address labels. To have a consistent output,
    // all those labels have to be switched to the generall "ADDRESS" label.
    google::protobuf::RepeatedPtrField<bert_annotator::LabeledSpan> empty_list;
    google::protobuf::RepeatedPtrField<bert_annotator::LabeledSpan>* const
        labeled_spans = GetLabelListWithDefault(&document, &empty_list);
    for (int i = labeled_spans->size() - 1; i >= 0; --i) {
      bert_annotator::LabeledSpan& labeled_span = labeled_spans->at(i);
      if (kAddressLabels.contains(labeled_span.label())) {
        labeled_span.set_label(
            std::string(Augmenter::kAddressReplacementLabel));

        // Consecutive address labels should be merged.
        if (i <= labeled_spans->size() - 2) {
          bert_annotator::LabeledSpan& successor_span =
              labeled_spans->at(i + 1);
          if (successor_span.label() == Augmenter::kAddressReplacementLabel) {
            labeled_span.set_token_end(successor_span.token_end());
            labeled_spans->erase(labeled_spans->begin() + i + 1);
          }
        }
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
      static_cast<double>(augmentations_.num_address_replacements) /
          augmentations_.num_total,
      address_sampler_, Augmenter::kAddressReplacementLabel,
      augmented_document);
  if (replaced_address) {
    --augmentations_.num_address_replacements;
    return true;
  }
  return false;
}

bool Augmenter::AugmentPhone(
    bert_annotator::Document* const augmented_document) {
  const bool replaced_phone = MaybeReplaceLabel(
      static_cast<double>(augmentations_.num_phone_replacements) /
          augmentations_.num_total,
      phone_sampler_, Augmenter::kPhoneReplacementLabel, augmented_document);
  if (replaced_phone) {
    --augmentations_.num_phone_replacements;
    return true;
  }
  return false;
}

bool Augmenter::AugmentLowercase(
    bert_annotator::Document* const augmented_document) {
  const bool perform_lowercasing = absl::Bernoulli(
      bitgenref_, static_cast<double>(augmentations_.num_lowercasings) /
                      augmentations_.num_total);
  if (perform_lowercasing) {
    Lowercase(augmented_document);
    --augmentations_.num_lowercasings;
    return true;
  }
  return false;
}

void Augmenter::AugmentContextless(const absl::string_view label,
                                   RandomSampler* const sampler) {
  bert_annotator::Document* document = documents_.add_documents();

  std::string sample = sampler->Sample();
  document->set_text(sample);
  auto token = document->add_token();
  token->set_word(sample);
  token->set_start(0);
  token->set_end(sample.size() - 1);
  bert_annotator::LabeledSpans labeled_spans = {};
  auto labeled_span = labeled_spans.add_labeled_span();
  labeled_span->set_label(std::string(label));
  labeled_span->set_token_start(0);
  labeled_span->set_token_end(0);
  (*document->mutable_labeled_spans())["lucid"] = labeled_spans;
}

void Augmenter::Augment() {
  const int original_document_number = documents_.documents_size();

  for (int i = 0; i < augmentations_.num_contextless_addresses; ++i) {
    AugmentContextless(kAddressReplacementLabel, address_sampler_);
    --augmentations_.num_total;
  }
  for (int i = 0; i < augmentations_.num_contextless_phones; ++i) {
    AugmentContextless(kPhoneReplacementLabel, phone_sampler_);
    --augmentations_.num_total;
  }

  while (augmentations_.num_total > 0) {
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

    if (augmentations_.mask_digits) {
      MaskDigits(augmented_document);
    }

    // If no action was performed and all remaining augmentations have to
    // perform at least one action, drop this sample. It's identical to the
    // original document. Repeat this augmentation iteration.
    if (!augmentation_performed &&
        augmentations_.num_total ==
            augmentations_.num_lowercasings +
                augmentations_.num_address_replacements +
                augmentations_.num_phone_replacements +
                augmentations_.num_context_drops_between_labels +
                augmentations_.num_context_drops_outside_one_label) {
      documents_.mutable_documents()->RemoveLast();
    } else {
      --augmentations_.num_total;
    }
  }
}

bool Augmenter::AugmentContext(
    bert_annotator::Document* const augmented_document) {
  const bool dropped_context_keeping_labels = MaybeDropContextKeepLabels(
      static_cast<double>(augmentations_.num_context_drops_between_labels) /
          augmentations_.num_total,
      augmented_document);
  if (dropped_context_keeping_labels) {
    --augmentations_.num_context_drops_between_labels;
    return true;
  }

  const bool dropped_context_dropping_labels = MaybeDropContextDropLabels(
      static_cast<double>(augmentations_.num_context_drops_outside_one_label) /
          augmentations_.num_total,
      augmented_document);
  if (dropped_context_dropping_labels) {
    --augmentations_.num_context_drops_outside_one_label;
    return true;
  }

  return false;
}

std::vector<TokenRange> Augmenter::GetUnlabeledRanges(
    const bert_annotator::Document& document) {
  std::vector<TokenRange> droppable_ranges;
  int start = 0;

  const google::protobuf::RepeatedPtrField<bert_annotator::LabeledSpan>&
      labeled_spans = GetLabelListWithDefault(document, {});
  for (const bert_annotator::LabeledSpan& labeled_span : labeled_spans) {
    if (labeled_span.label() == Augmenter::kAddressReplacementLabel ||
        labeled_span.label() == Augmenter::kPhoneReplacementLabel) {
      if (start != labeled_span.token_start()) {
        droppable_ranges.push_back(
            TokenRange{.start = start, .end = labeled_span.token_start() - 1});
      }
      start = labeled_span.token_end() + 1;
    }
  }
  if (start != document.token_size()) {
    droppable_ranges.push_back(
        TokenRange{.start = start, .end = document.token_size() - 1});
  }

  return droppable_ranges;
}

std::vector<TokenRange> Augmenter::GetLabeledRanges(
    const bert_annotator::Document& document,
    absl::flat_hash_set<absl::string_view> labels) {
  std::vector<TokenRange> labeled_ranges;
  const google::protobuf::RepeatedPtrField<bert_annotator::LabeledSpan>&
      labeled_spans = GetLabelListWithDefault(document, {});
  for (const bert_annotator::LabeledSpan& labeled_span : labeled_spans) {
    if (labels.contains(labeled_span.label())) {
      labeled_ranges.push_back(TokenRange{.start = labeled_span.token_start(),
                                          .end = labeled_span.token_end()});
    }
  }
  return labeled_ranges;
}

bool Augmenter::MaybeDropContextKeepLabels(
    const double probability,
    bert_annotator::Document* const augmented_document) {
  const bool do_drop_context = absl::Bernoulli(bitgenref_, probability);
  if (!do_drop_context) {
    return false;
  }

  bool dropped_context = false;
  std::vector<TokenRange> droppable_ranges =
      GetUnlabeledRanges(*augmented_document);
  // If there are no labels, we will get a single droppable range containing
  // the whole sentence. To drop from both its start and end, we duplicate it
  // here.
  bool no_labels = false;
  if (droppable_ranges.size() == 1 && droppable_ranges[0].start == 0 &&
      droppable_ranges[0].end == augmented_document->token_size() - 1) {
    no_labels = true;
    droppable_ranges.push_back(droppable_ranges[0]);
  }

  // Tokens will be dropped, so iterating backwards avoids the need to
  // update the indices in droppable_ranges.
  for (int i = droppable_ranges.size() - 1; i >= 0; --i) {
    const TokenRange& droppable_range = droppable_ranges[i];
    const bool do_drop =
        absl::Bernoulli(bitgenref_, augmentations_.probability_per_drop);
    if (!do_drop) {
      continue;
    }

    // At least one token should be dropped, so the range needs to be at
    // least two tokens long.
    if (droppable_range.end == droppable_range.start) {
      continue;
    }

    dropped_context = true;

    int drop_tokens_start;
    int drop_tokens_end;
    // For context after the last label, drop a postfix of the sentence.
    if (i == static_cast<int>(droppable_ranges.size()) - 1 &&
        droppable_range.end == augmented_document->token_size() - 1) {
      drop_tokens_start =
          absl::Uniform(absl::IntervalClosed, bitgenref_,
                        droppable_range.start + 1, droppable_range.end);
      drop_tokens_end = droppable_range.end;
      // If no labels exist, the range was duplicated, so the index needs to
      // be updated.
      if (no_labels) {
        droppable_ranges[0].end = drop_tokens_start;
      }
    }
    // For context before the first label, drop a prefix of the sentence. //
    else if (droppable_range.start == 0) {  // NOLINT
      drop_tokens_start = 0;
      drop_tokens_end = absl::Uniform(absl::IntervalClosed, bitgenref_, 0,
                                      droppable_range.end - 1);
    }
    // For context between two labels, drop a subsequence.
    else {  // NOLINT
      // At least the first and the last tokens must not be dropped.
      if (droppable_range.end - droppable_range.start == 1) {
        continue;
      }

      drop_tokens_start =
          absl::Uniform(absl::IntervalClosed, bitgenref_,
                        droppable_range.start + 1, droppable_range.end - 1);
      drop_tokens_end =
          absl::Uniform(absl::IntervalClosed, bitgenref_, drop_tokens_start,
                        droppable_range.end - 1);
    }

    const TokenRange boundaries =
        TokenRange{.start = drop_tokens_start, .end = drop_tokens_end};
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
  std::vector<TokenRange> labeled_ranges = GetLabeledRanges(
      *augmented_document, {kAddressReplacementLabel, kPhoneReplacementLabel});
  // MaybeDropContextKeepLabels already implements dropping from sentences
  // without any labels.
  if (labeled_ranges.size() == 0) {
    return MaybeDropContextKeepLabels(probability, augmented_document);
  }

  const bool do_drop_context = absl::Bernoulli(bitgenref_, probability);
  if (!do_drop_context || token_count == 1) {
    return false;
  }

  bool dropped_context = false;
  const int label_id =
      absl::Uniform(bitgenref_, 0, static_cast<int>(labeled_ranges.size()));
  TokenRange labeled_range = labeled_ranges[label_id];
  if (labeled_range.end < token_count - 2) {
    TokenRange drop_range{.start = 0, .end = token_count - 1};
    const bool drop_right =
        absl::Bernoulli(bitgenref_, augmentations_.probability_per_drop);
    if (drop_right) {
      drop_range.start = absl::Uniform(absl::IntervalClosed, bitgenref_,
                                       labeled_range.end + 2, token_count - 1);
      const int removed_characters = DropText(drop_range, augmented_document);
      DropTokens(drop_range, augmented_document);
      ShiftTokenBoundaries(drop_range.start, -removed_characters,
                           augmented_document);
      UpdateLabeledSpansForDroppedTokens(drop_range, augmented_document);
      dropped_context = true;
    }
  }
  if (labeled_range.start > 1) {
    TokenRange drop_range{.start = 0, .end = 0};
    const bool drop_left =
        absl::Bernoulli(bitgenref_, augmentations_.probability_per_drop);
    if (drop_left) {
      drop_range.end = absl::Uniform(absl::IntervalClosed, bitgenref_, 0,
                                     labeled_range.start - 2);
      const int removed_characters = DropText(drop_range, augmented_document);
      DropTokens(drop_range, augmented_document);
      ShiftTokenBoundaries(drop_range.start, -removed_characters,
                           augmented_document);
      UpdateLabeledSpansForDroppedTokens(drop_range, augmented_document);
      dropped_context = true;
    }
  }

  return dropped_context;
}

bool Augmenter::MaybeReplaceLabel(const double probability,
                                  RandomSampler* const sampler,
                                  const absl::string_view label,
                                  bert_annotator::Document* const document) {
  const std::vector<TokenRange>& boundary_list =
      GetLabeledRanges(*document, {label});
  const bool do_replace = absl::Bernoulli(bitgenref_, probability);
  if (do_replace && !boundary_list.empty()) {
    const int boundary_index =
        absl::Uniform(absl::IntervalClosed, bitgenref_, static_cast<size_t>(0),
                      boundary_list.size() - 1);
    const TokenRange boundaries = boundary_list[boundary_index];
    const std::string replacement = sampler->Sample();
    ReplaceLabeledTokens(boundaries, replacement, label, document);
    return true;
  }
  return false;
}

const int Augmenter::ReplaceText(
    const TokenRange& boundaries, const std::string& replacement,
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

const int Augmenter::DropText(const TokenRange& boundaries,
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
  bert_annotator::Token* token = document->mutable_token(token_id);
  token->set_word(replacement);
  token->set_end(token->start() + replacement.size() - 1);
}

void Augmenter::DropTokens(const TokenRange boundaries,
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
    bert_annotator::Token* token = document->mutable_token(i);
    token->set_start(token->start() + shift);
    token->set_end(token->end() + shift);
  }
}

void Augmenter::ReplaceLabeledSpan(
    const int token_id, const absl::string_view replacement_label,
    bert_annotator::Document* const document) const {
  google::protobuf::RepeatedPtrField<bert_annotator::LabeledSpan> empty_list;
  google::protobuf::RepeatedPtrField<bert_annotator::LabeledSpan>* const
      labeled_spans = GetLabelListWithDefault(document, &empty_list);
  for (bert_annotator::LabeledSpan& labeled_span : *labeled_spans) {
    if (labeled_span.token_start() == token_id) {
      labeled_span.set_label(std::string(replacement_label));
      labeled_span.set_token_end(labeled_span.token_start());
    }
  }
}

void Augmenter::UpdateLabeledSpansForDroppedTokens(
    const TokenRange& removed_tokens,
    bert_annotator::Document* const document) const {
  google::protobuf::RepeatedPtrField<bert_annotator::LabeledSpan> empty_list;
  google::protobuf::RepeatedPtrField<bert_annotator::LabeledSpan>* const
      labeled_spans = GetLabelListWithDefault(document, &empty_list);
  for (int i = labeled_spans->size() - 1; i >= 0; --i) {
    bert_annotator::LabeledSpan* labeled_span = labeled_spans->Mutable(i);
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
    const TokenRange& boundaries, const std::string& replacement,
    const absl::string_view replacement_label,
    bert_annotator::Document* const document) const {
  const int text_shift = ReplaceText(boundaries, replacement, document);
  ReplaceToken(boundaries.start, replacement, document);
  DropTokens(TokenRange{.start = boundaries.start + 1, .end = boundaries.end},
             document);
  ShiftTokenBoundaries(boundaries.start + 1, text_shift, document);
  ReplaceLabeledSpan(boundaries.start, replacement_label, document);
  UpdateLabeledSpansForDroppedTokens(
      TokenRange{.start = boundaries.start + 1, .end = boundaries.end},
      document);
}

void Augmenter::Lowercase(
    bert_annotator::Document* const augmented_document) const {
  std::string* const text = augmented_document->mutable_text();
  std::string new_text;
  int text_index = 0;
  for (bert_annotator::Token& token : *augmented_document->mutable_token()) {
    // Adds the string in between two tokens as it is.
    const int token_start = token.start();
    const int token_end = token.end();
    if (text_index < token_start) {
      new_text.append(text->begin() + text_index, text->begin() + token_start);
    }

    // Transforms the token to lowercase.
    std::string* const word = token.mutable_word();
    absl::AsciiStrToLower(word);
    new_text.append(*word);
    text_index = token_end + 1;
  }
  new_text.append(text->begin() + text_index, text->end());
  augmented_document->set_text(new_text);
}

google::protobuf::RepeatedPtrField<bert_annotator::LabeledSpan>
Augmenter::GetLabelListWithDefault(
    const bert_annotator::Document& document,
    google::protobuf::RepeatedPtrField<bert_annotator::LabeledSpan> defaults_to)
    const {
  if (document.labeled_spans().find("lucid") ==
      document.labeled_spans().end()) {
    return defaults_to;
  }

  return document.labeled_spans().at("lucid").labeled_span();
}

google::protobuf::RepeatedPtrField<bert_annotator::LabeledSpan>*
Augmenter::GetLabelListWithDefault(
    bert_annotator::Document* document,
    google::protobuf::RepeatedPtrField<bert_annotator::LabeledSpan>*
        defaults_to) const {
  if (document->labeled_spans().find("lucid") ==
      document->labeled_spans().end()) {
    return defaults_to;
  }

  return document->mutable_labeled_spans()->at("lucid").mutable_labeled_span();
}

void Augmenter::MaskDigits(std::string* text) const {
  absl::StrReplaceAll({{"1", "0"},
                       {"2", "0"},
                       {"3", "0"},
                       {"4", "0"},
                       {"5", "0"},
                       {"6", "0"},
                       {"7", "0"},
                       {"8", "0"},
                       {"9", "0"}},
                      text);
}

void Augmenter::MaskDigits(bert_annotator::Document* const document) const {
  MaskDigits(document->mutable_text());

  for (bert_annotator::Token& token : *document->mutable_token()) {
    MaskDigits(token.mutable_word());
  }
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
