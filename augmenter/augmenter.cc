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
#include "augmenter/random_sampler.h"
#include "protocol_buffer/document.pb.h"
#include "protocol_buffer/documents.pb.h"

namespace augmenter {

Augmenter::Augmenter(const bert_annotator::Documents documents,
                     RandomSampler* address_sampler,
                     RandomSampler* phones_sampler, absl::BitGenRef bitgenref)
    : documents_(documents),
      address_sampler_(address_sampler),
      phones_sampler_(phones_sampler),
      address_labels_({"LOCALITY", "COUNTRY", "ADMINISTRATIVE_AREA",
                       "THOROUGHFARE", "THOROUGHFARE_NUMBER", "PREMISE",
                       "POSTAL_CODE", "PREMISE_LEVEL"}),
      phone_labels_({"TELEPHONE"}),
      bitgenref_(bitgenref) {}

Augmenter::Augmenter(const bert_annotator::Documents documents,
                     RandomSampler* address_sampler,
                     RandomSampler* phones_sampler)
    : Augmenter(documents, address_sampler, phones_sampler, bitgen_) {}

void Augmenter::Augment(int total, int lowercase, int addresses, int phones) {
  const int original_document_number = documents_.documents_size();
  for (int i = 0; i < total; ++i) {
    int replacements = addresses + phones;
    const int remaining_total = total - i;
    bool augmentation_performed = false;

    const int document_id =
        absl::Uniform(bitgenref_, 0, original_document_number - 1);
    const bert_annotator::Document& original_document =
        documents_.documents(document_id);
    bert_annotator::Document* augmented_document = documents_.add_documents();
    augmented_document->CopyFrom(original_document);
    std::cout << "Next: " << original_document.text() << std::endl;

    auto address_boundary_list =
        DocumentBoundaryList(*augmented_document, address_labels_);
    if (!address_boundary_list.empty()) {
      std::cout << "Replace address" << std::endl;
      int address_boundary_index = absl::Uniform(
          bitgenref_, static_cast<size_t>(0), address_boundary_list.size() - 1);
      std::pair<int, int> address_boundaries =
          address_boundary_list[address_boundary_index];
      std::string replacement_address = address_sampler_->Sample();
      std::cout << "Replacing..." << std::endl;
      ReplaceTokens(augmented_document, address_boundaries,
                    replacement_address);
      std::cout << address_boundaries.first << " " << address_boundaries.second
                << std::endl;
      std::cout << original_document.text() << std::endl;
      std::cout << augmented_document->text() << std::endl << std::endl;
    }

    auto phone_boundary_list =
        DocumentBoundaryList(*augmented_document, phone_labels_);
    bool replace_phone =
        absl::Bernoulli(bitgenref_, static_cast<double>(phones) / total);
    if (replace_phone && !phone_boundary_list.empty()) {
      std::cout << "Replace phone" << std::endl;
      int phone_boundary_index = absl::Uniform(
          bitgenref_, static_cast<size_t>(0), phone_boundary_list.size() - 1);
      std::pair<int, int> phone_boundaries =
          phone_boundary_list[phone_boundary_index];
      std::string replacement_phone = phones_sampler_->Sample();
      ReplaceTokens(augmented_document, phone_boundaries, replacement_phone);
      std::cout << phone_boundaries.first << " " << phone_boundaries.second
                << std::endl;
      std::cout << original_document.text() << std::endl;
      std::cout << augmented_document->text() << std::endl << std::endl;
    }

    const bool perform_lowercasing = absl::Bernoulli(
        bitgenref_, static_cast<double>(lowercase) / remaining_total);
    if (perform_lowercasing) {
      Lowercase(augmented_document);
      --lowercase;
    }
  }
}

void Augmenter::ReplaceTokens(bert_annotator::Document* document,
                              std::pair<int, int> boundaries,
                              std::string replacement) {
  int address_string_start = document->token(boundaries.first).start();
  int address_string_end = document->token(boundaries.second).end();
  std::cout << "A..." << std::endl;

  // Replace the content of document->text().
  std::vector<char> new_text_bytes = std::vector<char>();
  std::cout << "A1..." << std::endl;
  new_text_bytes.insert(
      new_text_bytes.end(), document->mutable_text()->begin(),
      document->mutable_text()->begin() + address_string_start);
  std::cout << "A2..." << std::endl;
  new_text_bytes.insert(new_text_bytes.end(), replacement.begin(),
                        replacement.end());
  std::cout << "A3..." << std::to_string(document->text().size()) << " "
            << std::to_string(address_string_end) << std::endl;
  if (document->text().size() > address_string_end) {
    new_text_bytes.insert(
        new_text_bytes.end(),
        document->mutable_text()->begin() + address_string_end + 1,
        document->mutable_text()->end());
  }
  std::cout << "A4..." << std::endl;
  const std::string new_text(new_text_bytes.begin(), new_text_bytes.end());
  document->set_text(new_text);
  std::cout << "B..." << std::endl;

  // Replace the tokens. The first one summarizes the new content, all remaining
  // ones can be deleted. This introduces tokens longer than one word.
  std::cout << std::to_string(boundaries.first)
            << std::to_string(boundaries.second) << std::endl;

  document->mutable_token(boundaries.first)
      ->set_end(document->token(boundaries.second).end());
  document->mutable_token(boundaries.first)->set_word(replacement);
  if (boundaries.first != boundaries.second) {
    document->mutable_token()->erase(
        document->mutable_token()->begin() + boundaries.first + 1,
        document->mutable_token()->begin() + boundaries.second);
  }
  std::cout << "C..." << std::endl;

  // Update the start end end bytes of all tokens following the replaced
  // sequence.
  int length_increase =
      replacement.size() - (address_string_end - address_string_start);
  for (auto& token : *document->mutable_token()) {
    if (token.start() >= address_string_end) {
      token.set_start(token.start() + length_increase);
      token.set_end(token.end() + length_increase);
    }
  }
  std::cout << "D..." << std::endl;

  // Replace the labeled spans. The first one summarizes the new content, all
  // remaining ones can be deleted.
  int delete_start = 0;
  int delete_end = 0;
  auto labeled_spans =
      document->mutable_labeled_spans()->at("lucid").mutable_labeled_span();
  for (int i = 0; i < labeled_spans->size(); ++i) {
    auto labeled_span = labeled_spans->Get(i);
    if (labeled_span.token_start() == boundaries.first) {
      labeled_span.set_token_end(boundaries.second);
      labeled_span.set_label("ADDRESS");
      delete_start = i + 1;
    }
    if (labeled_span.token_start() <= boundaries.second) {
      delete_end = i + 1;
    }
  }
  std::cout << "E..." << std::endl;

  labeled_spans->erase(labeled_spans->begin() + delete_start,
                       labeled_spans->begin() + delete_end);
  std::cout << "F..." << std::endl;
}

std::vector<std::pair<int, int>> Augmenter::DocumentBoundaryList(
    const bert_annotator::Document& document,
    const std::vector<std::string>& labels) {
  auto labeled_spans = document.labeled_spans().at("lucid").labeled_span();
  std::cout << "0..." << std::endl;
  // First, select only spans labeled as an address. Then, join subsequent
  // spans.
  std::vector<std::pair<int, int>> boundary_list = {};
  for (int i = 0; i < labeled_spans.size(); ++i) {
    auto labeled_span = labeled_spans[i];
    if (std::find(std::begin(labels), std::end(labels), labeled_span.label()) !=
        std::end(labels)) {
      boundary_list.push_back(std::pair<int, int>(labeled_span.token_start(),
                                                  labeled_span.token_end()));
    }
  }
  std::cout << "1..." << std::endl;

  for (int i = boundary_list.size() - 2; i > 0; --i) {
    if (boundary_list[i].second + 1 == boundary_list[i + 1].first) {
      boundary_list[i].second = boundary_list[i + 1].second;
      boundary_list.erase(boundary_list.begin() + i + 1);
    }
  }
  std::cout << "2..." << std::endl;

  return boundary_list;
}

// Transforms the text to lowercase.
// Only explicitly listed tokens are transformed.
void Augmenter::Lowercase(bert_annotator::Document* const augmented_document) {
  std::string* text = augmented_document->mutable_text();
  std::vector<char> new_text_bytes = std::vector<char>();
  int text_index = 0;
  for (int j = 0; j < augmented_document->token_size(); ++j) {
    bert_annotator::Token* token = augmented_document->mutable_token(j);

    // Adds the string inbetween two tokens as it is.
    const int token_start = token->start();
    const int token_end = token->end();
    if (text_index < token_start) {
      new_text_bytes.insert(new_text_bytes.end(), text->begin() + text_index,
                            text->begin() + token_start);
    }

    // Transforms the token to lowercase.
    std::string* word = token->mutable_word();
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
