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

#include "gtest/gtest.h"
#include "protocol_buffer/document.pb.h"
#include "protocol_buffer/documents.pb.h"

TEST(AugmenterTest, AugmentsAreAdded) {
  bert_annotator::Documents documents;
  bert_annotator::Document* document = documents.add_documents();
  document->set_text("Text with some InterWordCapitalization");
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
  Augmenter augmenter = Augmenter(documents);

  augmenter.lowercase(Percentage(100));

  ASSERT_EQ(augmenter.get_documents().documents_size(), 2);
}

TEST(AugmenterTest, NoLowercasingForZeroPercent) {
  bert_annotator::Documents documents;
  bert_annotator::Document* document = documents.add_documents();
  document->set_text("Text with some InterWordCapitalization");
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
  Augmenter augmenter = Augmenter(documents);

  augmenter.lowercase(Percentage(0));

  ASSERT_STREQ(augmenter.get_documents().documents(0).text().c_str(),
               "Text with some InterWordCapitalization");
}

TEST(AugmenterTest, CompleteLowercasingForHundretPercent) {
  bert_annotator::Documents documents;
  bert_annotator::Document* document = documents.add_documents();
  document->set_text("Text with some InterWordCapitalization");
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
  Augmenter augmenter = Augmenter(documents);

  augmenter.lowercase(Percentage(100));

  ASSERT_STREQ(augmenter.get_documents().documents(1).text().c_str(),
               "text with some interwordcapitalization");
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

  augmenter.lowercase(Percentage(100));

  ASSERT_STREQ(augmenter.get_documents().documents(1).text().c_str(),
               "[BOS] text with some interwordcapitalization [EOS]");
}
