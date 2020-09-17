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

#include "gtest/gtest.h"
#include "protocol_buffer/document.pb.h"
#include "protocol_buffer/documents.pb.h"

namespace augmenter {

struct TokenSpec {
  const std::string text;
  const int start;
  const int end;
  TokenSpec(const std::string text_, const int start_, const int end_)
      : text(text_), start(start_), end(end_) {}
};

struct DocumentSpec {
  const std::string text;
  const std::vector<TokenSpec> token_specs;
  DocumentSpec(const std::string text_,
               const std::vector<TokenSpec> token_specs_)
      : text(text_), token_specs(token_specs_) {}
};

// Creates a documents wrapper with documents each containing the specified
// string and list of tokens.
bert_annotator::Documents ConstructBertDocument(
    const std::vector<DocumentSpec> document_specs) {
  bert_annotator::Documents documents = bert_annotator::Documents();
  for (const DocumentSpec document_spec : document_specs) {
    bert_annotator::Document* document = documents.add_documents();
    document->set_text(document_spec.text);
    for (const TokenSpec token_spec : document_spec.token_specs) {
      bert_annotator::Token* token;
      token = document->add_token();
      token->set_start(token_spec.start);
      token->set_end(token_spec.end);
      token->set_word(token_spec.text);
    }
  }

  return documents;
}

TEST(AugmenterTest, AugmentsAreAdded) {
  bert_annotator::Documents documents = ConstructBertDocument(
      {DocumentSpec("Text with some InterWordCapitalization", {})});
  Augmenter augmenter = Augmenter(documents);

  augmenter.Lowercase(/*lowercase_percentage=*/1.0);

  ASSERT_EQ(augmenter.documents().documents_size(), 2);
}

TEST(AugmenterTest, NoLowercasingForZeroPercent) {
  bert_annotator::Documents documents = ConstructBertDocument(
      {DocumentSpec("Text with some InterWordCapitalization", {})});
  Augmenter augmenter = Augmenter(documents);

  augmenter.Lowercase(/*lowercase_percentage=*/0.0);

  ASSERT_STREQ(augmenter.documents().documents(0).text().c_str(),
               "Text with some InterWordCapitalization");
}

TEST(AugmenterTest, CompleteLowercasingForHundredPercent) {
  bert_annotator::Documents documents = ConstructBertDocument(
      {DocumentSpec("Text with some InterWordCapitalization",
                    {TokenSpec("Text", 0, 3), TokenSpec("with", 5, 8),
                     TokenSpec("some", 10, 13),
                     TokenSpec("InterWordCapitalization", 15, 37)})});
  Augmenter augmenter = Augmenter(documents);

  augmenter.Lowercase(/*lowercase_percentage=*/1.0);

  ASSERT_STREQ(augmenter.documents().documents(1).text().c_str(),
               "text with some interwordcapitalization");
}

TEST(AugmenterTest, RandomizedLowercasing) {
  bert_annotator::Documents documents = ConstructBertDocument(
      {DocumentSpec("Text with some InterWordCapitalization [0]",
                    {TokenSpec("Text", 0, 3), TokenSpec("with", 5, 8),
                     TokenSpec("some", 10, 13),
                     TokenSpec("InterWordCapitalization", 15, 37)}),
       DocumentSpec("Text with some InterWordCapitalization [1]",
                    {TokenSpec("Text", 0, 3), TokenSpec("with", 5, 8),
                     TokenSpec("some", 10, 13),
                     TokenSpec("InterWordCapitalization", 15, 37)})});
  Augmenter augmenter = Augmenter(documents, /*seed=*/0);

  augmenter.Lowercase(/*lowercase_percentage=*/0.5);

  ASSERT_EQ(augmenter.documents().documents_size(), 3);
  EXPECT_STREQ(augmenter.documents().documents(0).text().c_str(),
               "Text with some InterWordCapitalization [0]");
  EXPECT_STREQ(augmenter.documents().documents(1).text().c_str(),
               "Text with some InterWordCapitalization [1]");
  EXPECT_STREQ(augmenter.documents().documents(2).text().c_str(),
               "text with some interwordcapitalization [0]");
}

TEST(AugmenterTest, DontLowercaseNonTokens) {
  bert_annotator::Documents documents = ConstructBertDocument(
      {DocumentSpec("[BOS] Text with some InterWordCapitalization [EOS]",
                    {TokenSpec("Text", 6, 9), TokenSpec("with", 11, 14),
                     TokenSpec("some", 16, 19),
                     TokenSpec("InterWordCapitalization", 21, 43)})});
  Augmenter augmenter = Augmenter(documents);

  augmenter.Lowercase(/*lowercase_percentage=*/1.0);

  ASSERT_STREQ(augmenter.documents().documents(1).text().c_str(),
               "[BOS] text with some interwordcapitalization [EOS]");
}

}  // namespace augmenter
