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
#include <vector>

#include "absl/strings/ascii.h"
#include "protocol_buffer/document.pb.h"
#include "protocol_buffer/documents.pb.h"

namespace augmenter {

Augmenter::Augmenter(const bert_annotator::Documents documents,
                     absl::BitGenRef bitgenref)
    : documents_(documents), bitgenref_(bitgenref) {}

Augmenter::Augmenter(const bert_annotator::Documents documents)
    : Augmenter(documents, bitgen_) {}

void Augmenter::Augment(const int augmentations,
                        const double lowercase_percentage) {
  const int original_document_number = documents_.documents_size();
  int remaining_lowercase_augmentations = augmentations * lowercase_percentage;
  for (int i = 0; i < augmentations; ++i) {
    const int remaining_augmentations = augmentations - i;
    const int document_id = i % original_document_number;
    const bert_annotator::Document& original_document =
        documents_.documents(document_id);

    bert_annotator::Document* augmented_document = documents_.add_documents();
    augmented_document->CopyFrom(original_document);

    const bool perform_lowercasing = absl::Bernoulli(
        bitgenref_, static_cast<double>(remaining_lowercase_augmentations) /
                        remaining_augmentations);
    if (perform_lowercasing) {
      Lowercase(augmented_document);
      --remaining_lowercase_augmentations;
    }
  }
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
