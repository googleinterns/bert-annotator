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

void add_test_document(bert_annotator::Documents* documents,
                       int document_number) {
  bert_annotator::Document* document = documents->add_documents();
  document->set_text("Text with some InterWordCapitalization [" +
                     std::to_string(document_number) + "]");
  bert_annotator::Token* token;
  token = document->add_token();
  token->set_start(0);
  token->set_end(3);
  token->set_word("Text");
  token = document->add_token();
  token->set_start(5);
  token->set_end(8);
  token->set_word("with");
  token = document->add_token();
  token->set_start(10);
  token->set_end(13);
  token->set_word("some");
  token = document->add_token();
  token->set_start(15);
  token->set_end(37);
  token->set_word("InterWordCapitalization");
}

bert_annotator::Documents construct_test_documents(int document_number) {
  bert_annotator::Documents documents;
  for (int i = 0; i < document_number; ++i) {
    add_test_document(&documents, i);
  }

  return documents;
}

TEST(AugmenterTest, AugmentsAreAdded) {
  bert_annotator::Documents documents = construct_test_documents(1);
  Augmenter augmenter = Augmenter(documents);

  augmenter.lowercase(1.0);

  ASSERT_EQ(augmenter.get_documents().documents_size(), 2);
}

TEST(AugmenterTest, NoLowercasingForZeroPercent) {
  bert_annotator::Documents documents = construct_test_documents(1);
  Augmenter augmenter = Augmenter(documents);

  augmenter.lowercase(0.0);

  ASSERT_STREQ(augmenter.get_documents().documents(0).text().c_str(),
               "Text with some InterWordCapitalization [0]");
}

TEST(AugmenterTest, CompleteLowercasingForHundredPercent) {
  bert_annotator::Documents documents = construct_test_documents(1);
  Augmenter augmenter = Augmenter(documents);

  augmenter.lowercase(1.0);

  ASSERT_STREQ(augmenter.get_documents().documents(1).text().c_str(),
               "text with some interwordcapitalization [0]");
}

TEST(AugmenterTest, RandomizedLowercasing) {
  bert_annotator::Documents documents = construct_test_documents(2);
  Augmenter augmenter = Augmenter(documents, 0);

  augmenter.lowercase(0.5);

  ASSERT_EQ(augmenter.get_documents().documents_size(), 3);
  EXPECT_STREQ(augmenter.get_documents().documents(0).text().c_str(),
               "Text with some InterWordCapitalization [0]");
  EXPECT_STREQ(augmenter.get_documents().documents(1).text().c_str(),
               "Text with some InterWordCapitalization [1]");
  EXPECT_STREQ(augmenter.get_documents().documents(2).text().c_str(),
               "text with some interwordcapitalization [0]");
}

TEST(AugmenterTest, DontLowercaseNonTokens) {
  bert_annotator::Documents documents;
  bert_annotator::Document* document = documents.add_documents();
  document->set_text("[BOS] Text with some InterWordCapitalization [EOS]");
  bert_annotator::Token* token;
  token = document->add_token();
  token->set_start(6);
  token->set_end(9);
  token->set_word("Text");
  token = document->add_token();
  token->set_start(11);
  token->set_end(14);
  token->set_word("with");
  token = document->add_token();
  token->set_start(16);
  token->set_end(19);
  token->set_word("some");
  token = document->add_token();
  token->set_start(21);
  token->set_end(43);
  token->set_word("InterWordCapitalization");
  Augmenter augmenter = Augmenter(documents);

  augmenter.lowercase(1.0);

  ASSERT_STREQ(augmenter.get_documents().documents(1).text().c_str(),
               "[BOS] text with some interwordcapitalization [EOS]");
}

}  // namespace augmenter
