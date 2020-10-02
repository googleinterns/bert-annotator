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
#include "augmenter/label_boundaries.h"
#include "augmenter/random_sampler.h"
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

bool Augmenter::MaybeReplaceLabel(const double probability,
                                  RandomSampler* const sampler,
                                  const absl::string_view label,
                                  bert_annotator::Document* const document) {
  const std::vector<LabelBoundaries>& boundary_list =
      LabelBoundaryList(*document, label);
  const bool do_replace = absl::Bernoulli(bitgenref_, probability);
  if (do_replace && !boundary_list.empty()) {
    const int boundary_index = absl::Uniform(bitgenref_, static_cast<size_t>(0),
                                             boundary_list.size() - 1);
    const LabelBoundaries boundaries = boundary_list[boundary_index];
    const std::string replacement = sampler->Sample();
    Replace(boundaries, replacement, label, document);
    return true;
  }
  return false;
}

void Augmenter::ReplaceText(const LabelBoundaries& boundaries,
                            const std::string& replacement,
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
}

void Augmenter::ReplaceTokens(const LabelBoundaries& boundaries,
                              const std::string& replacement,
                              bert_annotator::Document* const document) const {
  document->mutable_token(boundaries.start)
      ->set_end(document->token(boundaries.end).end());
  document->mutable_token(boundaries.start)->set_word(replacement);
  if (boundaries.start != boundaries.end) {
    document->mutable_token()->erase(
        document->mutable_token()->begin() + boundaries.start + 1,
        document->mutable_token()->begin() + boundaries.end +
            1);  // boundaries.end is inclusive.
  }
}

void Augmenter::UpdateTokenBoundaries(
    const LabelBoundaries& boundaries, const std::string& replacement,
    bert_annotator::Document* const document) const {
  const int string_start = document->token(boundaries.start).start();
  const int string_end = document->token(boundaries.start).end();
  const int length_increase =
      replacement.size() - (string_end - string_start + 1);
  for (auto& token : *document->mutable_token()) {
    if (token.start() > string_start) {
      token.set_start(token.start() + length_increase);
    }
    if (token.end() >= string_end) {
      token.set_end(token.end() + length_increase);
    }
  }
}

void Augmenter::ReplaceLabeledSpans(
    const LabelBoundaries& boundaries,
    const absl::string_view replacement_label,
    bert_annotator::Document* const document) const {
  int delete_start = 0;
  int delete_end = 0;
  auto labeled_spans =
      document->mutable_labeled_spans()->at("lucid").mutable_labeled_span();
  int label_count_decrease = boundaries.end - boundaries.start;
  for (int i = 0; i < labeled_spans->size(); ++i) {
    auto labeled_span = labeled_spans->Mutable(i);
    if (labeled_span->token_start() == boundaries.start) {
      labeled_span->set_label(std::string(replacement_label));
      delete_start = i + 1;
    }
    if (labeled_span->token_start() <= boundaries.end) {
      delete_end = i + 1;
    }
    if (labeled_span->token_start() > boundaries.start) {
      labeled_span->set_token_start(labeled_span->token_start() -
                                    label_count_decrease);
    }
    if (labeled_span->token_end() >= boundaries.end) {
      labeled_span->set_token_end(labeled_span->token_end() -
                                  label_count_decrease);
    }
  }
  labeled_spans->erase(
      labeled_spans->begin() + delete_start,
      labeled_spans->begin() + delete_end);  // delete_end is exclusive.
}

void Augmenter::Replace(const LabelBoundaries& boundaries,
                        const std::string& replacement,
                        const absl::string_view replacement_label,
                        bert_annotator::Document* const document) const {
  ReplaceText(boundaries, replacement, document);
  ReplaceTokens(boundaries, replacement, document);
  UpdateTokenBoundaries(boundaries, replacement, document);
  ReplaceLabeledSpans(boundaries, replacement_label, document);
}

const std::vector<LabelBoundaries> Augmenter::LabelBoundaryList(
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
  std::vector<LabelBoundaries> boundary_list = {};
  for (int i = 0; i < labeled_spans.size(); ++i) {
    const auto labeled_span = labeled_spans[i];
    if (labeled_span.label().compare(std::string(label)) == 0) {
      boundary_list.push_back(
          LabelBoundaries{.start = labeled_span.token_start(),
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
