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

Augmenter::Augmenter(const bert_annotator::Documents documents)
    : documents_(documents), seed_(time(NULL)) {}

Augmenter::Augmenter(const bert_annotator::Documents documents, const uint seed)
    : documents_(documents), seed_(seed) {}

// Transforms the text to lowercase.
// Only explicitly listed tokens are transformed.
void Augmenter::Lowercase(const double lowercase_percentage) {
  const int num_original_documents = documents_.documents_size();
  for (int i = 0; i < num_original_documents; ++i) {
    // Skip if not in interval (0, 1].
    if (lowercase_percentage < (rand_r(&seed_) + 1.) / (RAND_MAX + 1.)) {
      continue;
    }

    const bert_annotator::Document& original_document = documents_.documents(i);

    bert_annotator::Document* augmented_document = documents_.add_documents();
    augmented_document->CopyFrom(original_document);

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
}

const bert_annotator::Documents Augmenter::documents() const {
  return documents_;
}

}  // namespace augmenter
