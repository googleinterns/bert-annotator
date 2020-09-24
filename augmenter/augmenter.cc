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
#include <utility>
#include <vector>

#include "absl/strings/ascii.h"
#include "augmenter/label_boundaries.h"
#include "augmenter/random_sampler.h"
#include "protocol_buffer/document.pb.h"
#include "protocol_buffer/documents.pb.h"

namespace augmenter {

Augmenter::Augmenter(const bert_annotator::Documents& documents,
                     RandomSampler* const address_sampler,
                     RandomSampler* const phone_sampler,
                     absl::BitGenRef bitgenref)
    : documents_(documents),
      address_sampler_(address_sampler),
      phone_sampler_(phone_sampler),
      address_labels_({"LOCALITY", "COUNTRY", "ADMINISTRATIVE_AREA",
                       "THOROUGHFARE", "THOROUGHFARE_NUMBER", "PREMISE",
                       "POSTAL_CODE", "PREMISE_LEVEL"}),
      address_replacement_label_("ADDRESS"),
      phone_labels_({"TELEPHONE"}),
      phone_replacement_label_("TELEPHONE"),
      bitgenref_(bitgenref) {}

Augmenter::Augmenter(const bert_annotator::Documents& documents,
                     RandomSampler* const address_sampler,
                     RandomSampler* const phone_sampler)
    : Augmenter(documents, address_sampler, phone_sampler, bitgen_) {}

void Augmenter::Augment(int augmentations_total, int augmentations_lowercase,
                        int augmentations_address, int augmentations_phone) {
  const int original_document_number = documents_.documents_size();
  for (int i = 0; i < augmentations_total; ++i) {
    const int augmentations_remaining_total = augmentations_total - i;
    bool augmentation_performed = false;

    const int document_id =
        absl::Uniform(bitgenref_, 0, original_document_number - 1);
    const bert_annotator::Document& original_document =
        documents_.documents(document_id);
    bert_annotator::Document* augmented_document = documents_.add_documents();
    augmented_document->CopyFrom(original_document);

    const bool replaced_address =
        MaybeReplaceLabel(augmented_document, address_labels_,
                          static_cast<double>(augmentations_address) /
                              augmentations_remaining_total,
                          address_sampler_, address_replacement_label_);
    if (replaced_address) {
      augmentation_performed = true;
      --augmentations_address;
    }

    const bool replaced_phone =
        MaybeReplaceLabel(augmented_document, phone_labels_,
                          static_cast<double>(augmentations_phone) /
                              augmentations_remaining_total,
                          phone_sampler_, phone_replacement_label_);
    if (replaced_phone) {
      augmentation_performed = true;
      --augmentations_phone;
    }

    const bool perform_lowercasing = absl::Bernoulli(
        bitgenref_, static_cast<double>(augmentations_lowercase) /
                        augmentations_remaining_total);
    if (perform_lowercasing) {
      Lowercase(augmented_document);
      --augmentations_lowercase;
      augmentation_performed = true;
    }

    // If no action was performed and all remaining augmentations have to
    // perform at least one action, drop this sample. It's identical to the
    // original document. Repeat this augmentation iteration.
    if (!augmentation_performed &&
        augmentations_remaining_total == augmentations_lowercase +
                                             augmentations_address +
                                             augmentations_phone) {
      documents_.mutable_documents()->RemoveLast();
      --i;
    }
  }
}

bool Augmenter::MaybeReplaceLabel(bert_annotator::Document* const document,
                                  const std::vector<std::string>& label_list,
                                  const double probability,
                                  RandomSampler* const sampler,
                                  const std::string& replacement_label) {
  const std::vector<LabelBoundaries>& boundary_list =
      LabelBoundaryList(*document, label_list);
  const bool do_replace = absl::Bernoulli(bitgenref_, probability);
  if (do_replace && !boundary_list.empty()) {
    const int boundary_index = absl::Uniform(bitgenref_, static_cast<size_t>(0),
                                             boundary_list.size() - 1);
    const LabelBoundaries boundaries = boundary_list[boundary_index];
    const std::string replacement = sampler->Sample();
    ReplaceTokens(document, boundaries, replacement, replacement_label);
    return true;
  }
  return false;
}

void Augmenter::ReplaceTokens(bert_annotator::Document* const document,
                              const LabelBoundaries& boundaries,
                              const std::string& replacement,
                              const std::string& replacement_label) const {
  const int string_start = document->token(boundaries.start).start();
  const int string_end = document->token(boundaries.end).end();

  // Replace the content of document->text().
  std::vector<char> new_text_bytes = std::vector<char>();
  new_text_bytes.insert(new_text_bytes.end(), document->mutable_text()->begin(),
                        document->mutable_text()->begin() + string_start);
  new_text_bytes.insert(new_text_bytes.end(), replacement.begin(),
                        replacement.end());
  if (static_cast<int>(document->text().size()) > string_end) {
    new_text_bytes.insert(new_text_bytes.end(),
                          document->mutable_text()->begin() + string_end + 1,
                          document->mutable_text()->end());
  }
  const std::string new_text(new_text_bytes.begin(), new_text_bytes.end());
  document->set_text(new_text);

  // Replace the tokens. The first one summarizes the new content, all remaining
  // ones can be deleted. This introduces tokens longer than one word.
  document->mutable_token(boundaries.start)
      ->set_end(document->token(boundaries.end).end());
  document->mutable_token(boundaries.start)->set_word(replacement);
  if (boundaries.start != boundaries.end) {
    document->mutable_token()->erase(
        document->mutable_token()->begin() + boundaries.start + 1,
        document->mutable_token()->begin() + boundaries.end +
            1);  // boundaries.end is inclusive.
  }

  // Update the start end end bytes of all tokens following the replaced
  // sequence.
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

  // Replace the labeled spans. The first one summarizes the new content, all
  // remaining ones can be deleted.
  int delete_start = 0;
  int delete_end = 0;
  auto labeled_spans =
      document->mutable_labeled_spans()->at("lucid").mutable_labeled_span();
  int label_count_decrease = boundaries.end - boundaries.start;
  for (int i = 0; i < labeled_spans->size(); ++i) {
    auto labeled_span = labeled_spans->Mutable(i);
    if (labeled_span->token_start() == boundaries.start) {
      labeled_span->set_label(replacement_label);
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

const std::vector<LabelBoundaries> Augmenter::LabelBoundaryList(
    const bert_annotator::Document& document,
    const std::vector<std::string>& labels) const {
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
    if (std::find(labels.begin(), labels.end(), labeled_span.label()) !=
        labels.end()) {
      boundary_list.push_back(LabelBoundaries(labeled_span.token_start(),
                                              labeled_span.token_end()));
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

// Transforms the text to lowercase.
// Only explicitly listed tokens are transformed.
void Augmenter::Lowercase(
    bert_annotator::Document* const augmented_document) const {
  std::string* const text = augmented_document->mutable_text();
  std::vector<char> new_text_bytes = std::vector<char>();
  int text_index = 0;
  for (int j = 0; j < augmented_document->token_size(); ++j) {
    bert_annotator::Token* const token = augmented_document->mutable_token(j);

    // Adds the string inbetween two tokens as it is.
    const int token_start = token->start();
    const int token_end = token->end();
    if (text_index < token_start) {
      new_text_bytes.insert(new_text_bytes.end(), text->begin() + text_index,
                            text->begin() + token_start);
    }

    // Transforms the token to lowercase.
    std::string* const word = token->mutable_word();
    absl::AsciiStrToLower(word);
    new_text_bytes.insert(new_text_bytes.end(), word->begin(), word->end());
    text_index = token_end + 1;
  }
  new_text_bytes.insert(new_text_bytes.end(), text->begin() + text_index,
                        text->end());
  const std::string new_text(new_text_bytes.begin(), new_text_bytes.end());
  augmented_document->set_text(new_text);
}

const bert_annotator::Documents Augmenter::documents() const {
  return documents_;
}

}  // namespace augmenter
