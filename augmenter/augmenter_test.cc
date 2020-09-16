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

#include "gtest/gtest.h"
#include "protocol_buffer/document.pb.h"
#include "protocol_buffer/documents.pb.h"

namespace augmenter {

struct TestToken {
  const std::string text;
  const int start;
  const int end;
  TestToken(const std::string text_, const int start_, const int end_)
      : text(text_), start(start_), end(end_) {}
};

struct TestDocument {
  const std::string text;
  const std::vector<TestToken> test_tokens;
  TestDocument(const std::string text_,
               const std::vector<TestToken> test_tokens_)
      : text(text_), test_tokens(test_tokens_) {}
};

bert_annotator::Documents construct_test_documents(
    const std::vector<TestDocument> test_documents) {
  bert_annotator::Documents documents = bert_annotator::Documents();
  for (const TestDocument test_document : test_documents) {
    bert_annotator::Document* document = documents.add_documents();
    document->set_text(test_document.text);
    for (const TestToken test_token : test_document.test_tokens) {
      bert_annotator::Token* token;
      token = document->add_token();
      token->set_start(test_token.start);
      token->set_end(test_token.end);
      token->set_word(test_token.text);
    }
  }

  return documents;
}

TEST(AugmenterTest, AugmentsAreAdded) {
  bert_annotator::Documents documents = construct_test_documents(
      {TestDocument("Text with some InterWordCapitalization", {})});
  Augmenter augmenter = Augmenter(documents);

  augmenter.lowercase(/*lowercase_percentage=*/1.0);

  ASSERT_EQ(augmenter.get_documents().documents_size(), 2);
}

TEST(AugmenterTest, NoLowercasingForZeroPercent) {
  bert_annotator::Documents documents = construct_test_documents(
      {TestDocument("Text with some InterWordCapitalization", {})});
  Augmenter augmenter = Augmenter(documents);

  augmenter.lowercase(/*lowercase_percentage=*/0.0);

  ASSERT_STREQ(augmenter.get_documents().documents(0).text().c_str(),
               "Text with some InterWordCapitalization");
}

TEST(AugmenterTest, CompleteLowercasingForHundredPercent) {
  bert_annotator::Documents documents = construct_test_documents(
      {TestDocument("Text with some InterWordCapitalization",
                    {TestToken("Text", 0, 3), TestToken("with", 5, 8),
                     TestToken("some", 10, 13),
                     TestToken("InterWordCapitalization", 15, 37)})});
  Augmenter augmenter = Augmenter(documents);

  augmenter.lowercase(/*lowercase_percentage=*/1.0);

  ASSERT_STREQ(augmenter.get_documents().documents(1).text().c_str(),
               "text with some interwordcapitalization");
}

TEST(AugmenterTest, RandomizedLowercasing) {
  bert_annotator::Documents documents = construct_test_documents(
      {TestDocument("Text with some InterWordCapitalization [0]",
                    {TestToken("Text", 0, 3), TestToken("with", 5, 8),
                     TestToken("some", 10, 13),
                     TestToken("InterWordCapitalization", 15, 37)}),
       TestDocument("Text with some InterWordCapitalization [1]",
                    {TestToken("Text", 0, 3), TestToken("with", 5, 8),
                     TestToken("some", 10, 13),
                     TestToken("InterWordCapitalization", 15, 37)})});
  Augmenter augmenter = Augmenter(documents, /*seed=*/0);

  augmenter.lowercase(/*lowercase_percentage=*/0.5);

  ASSERT_EQ(augmenter.get_documents().documents_size(), 3);
  EXPECT_STREQ(augmenter.get_documents().documents(0).text().c_str(),
               "Text with some InterWordCapitalization [0]");
  EXPECT_STREQ(augmenter.get_documents().documents(1).text().c_str(),
               "Text with some InterWordCapitalization [1]");
  EXPECT_STREQ(augmenter.get_documents().documents(2).text().c_str(),
               "text with some interwordcapitalization [0]");
}

TEST(AugmenterTest, DontLowercaseNonTokens) {
  bert_annotator::Documents documents = construct_test_documents(
      {TestDocument("[BOS] Text with some InterWordCapitalization [EOS]",
                    {TestToken("Text", 6, 9), TestToken("with", 11, 14),
                     TestToken("some", 16, 19),
                     TestToken("InterWordCapitalization", 21, 43)})});
  Augmenter augmenter = Augmenter(documents);

  augmenter.lowercase(/*lowercase_percentage=*/1.0);

  ASSERT_STREQ(augmenter.get_documents().documents(1).text().c_str(),
               "[BOS] text with some interwordcapitalization [EOS]");
}

}  // namespace augmenter
