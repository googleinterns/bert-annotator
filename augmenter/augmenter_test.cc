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

#include <map>
#include <string>
#include <vector>

#include "absl/random/mock_distributions.h"
#include "augmenter/augmentations.h"
#include "gtest/gtest.h"
#include "protocol_buffer/document.pb.h"
#include "protocol_buffer/documents.pb.h"

namespace augmenter {

using ::testing::InSequence;
using ::testing::Return;
using ::testing::ReturnRef;

struct TokenSpec {
  const std::string& text;
  const int start;
  const int end;
  TokenSpec(const std::string& text_, const int start_, const int end_)
      : text(text_), start(start_), end(end_) {}
};

struct LabelSpec {
  const std::string& label;
  const int start;
  const int end;
  LabelSpec(const std::string& label_, const int start_, const int end_)
      : label(label_), start(start_), end(end_) {}
};

struct DocumentSpec {
  const std::string& text;
  const std::vector<TokenSpec>& token_specs;
  const bool has_labels;
  const std::map<std::string, std::vector<LabelSpec>>& label_specs_map;

  DocumentSpec(const std::string& text_,
               const std::vector<TokenSpec> token_specs_,
               const std::map<std::string, std::vector<LabelSpec>>
                   label_specs_map_ = {{"lucid", {}}})
      : text(text_),
        token_specs(token_specs_),
        has_labels(true),
        label_specs_map(label_specs_map_) {}

  DocumentSpec(
      const std::string& text_, const std::vector<TokenSpec> token_specs_,
      const std::map<std::string, std::vector<LabelSpec>> label_specs_map_,
      bool has_labels_)
      : text(text_),
        token_specs(token_specs_),
        has_labels(has_labels_),
        label_specs_map(label_specs_map_) {}
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

    if (document_spec.has_labels) {
      std::map<std::string, bert_annotator::LabeledSpans> label_map = {};
      bert_annotator::LabeledSpans labeled_spans = {};
      for (auto& label_pair : document_spec.label_specs_map) {
        for (auto& label_spec : label_pair.second) {
          auto labeled_span = labeled_spans.add_labeled_span();
          labeled_span->set_label(label_spec.label);
          labeled_span->set_token_start(label_spec.start);
          labeled_span->set_token_end(label_spec.end);
        }
        (*document->mutable_labeled_spans())[label_pair.first] = labeled_spans;
      }
    }
  }

  return documents;
}

void ExpectEq(const bert_annotator::Document a,
              const bert_annotator::Document b) {
  EXPECT_STREQ(a.text().c_str(), b.text().c_str());
  ASSERT_EQ(a.token_size(), b.token_size());
  for (int i = 0; i < a.token_size(); ++i) {
    EXPECT_STREQ(a.token(i).word().c_str(), b.token(i).word().c_str());
    EXPECT_EQ(a.token(i).start(), b.token(i).start());
    EXPECT_EQ(a.token(i).end(), b.token(i).end());
  }
  ASSERT_EQ(a.labeled_spans_size(), b.labeled_spans_size());
  for (auto map_entry : a.labeled_spans()) {
    auto key = map_entry.first;
    ASSERT_EQ(a.labeled_spans().at(key).labeled_span_size(),
              b.labeled_spans().at(key).labeled_span_size());
    for (int i = 0; i < a.labeled_spans().at(key).labeled_span_size(); ++i) {
      EXPECT_STREQ(a.labeled_spans().at(key).labeled_span(i).label().c_str(),
                   b.labeled_spans().at(key).labeled_span(i).label().c_str());
      EXPECT_EQ(a.labeled_spans().at(key).labeled_span(i).token_start(),
                b.labeled_spans().at(key).labeled_span(i).token_start());
      EXPECT_EQ(a.labeled_spans().at(key).labeled_span(i).token_end(),
                b.labeled_spans().at(key).labeled_span(i).token_end());
    }
  }
}

TEST(AugmenterTest, NoAugmentation) {
  bert_annotator::Documents documents = ConstructBertDocument(
      {DocumentSpec("Text with some InterWordCapitalization", {})});
  Augmentations augmentations = {
      .total = 0, .lowercase = 0, .address = 0, .phone = 0};
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  Augmenter augmenter =
      Augmenter(documents, augmentations, &address_sampler, &phone_sampler);

  augmenter.Augment();

  EXPECT_EQ(augmenter.documents().documents_size(), 1);
}

TEST(AugmenterTest, AugmentsAreAdded) {
  bert_annotator::Documents documents = ConstructBertDocument(
      {DocumentSpec("Text with some InterWordCapitalization", {})});
  Augmentations augmentations = {
      .total = 1, .lowercase = 0, .address = 0, .phone = 0};
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  Augmenter augmenter =
      Augmenter(documents, augmentations, &address_sampler, &phone_sampler);

  augmenter.Augment();

  EXPECT_EQ(augmenter.documents().documents_size(), 2);
}

TEST(AugmenterTest, NoLowercasing) {
  bert_annotator::Documents documents = ConstructBertDocument(
      {DocumentSpec("Text with some InterWordCapitalization", {})});
  Augmentations augmentations = {
      .total = 10, .lowercase = 0, .address = 0, .phone = 0};
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  Augmenter augmenter =
      Augmenter(documents, augmentations, &address_sampler, &phone_sampler);

  augmenter.Augment();

  for (int i = 0; i < augmentations.total + 1; ++i) {
    EXPECT_STREQ(augmenter.documents().documents(i).text().c_str(),
                 "Text with some InterWordCapitalization");
  }
}

TEST(AugmenterTest, CompleteLowercasing) {
  bert_annotator::Documents documents = ConstructBertDocument(
      {DocumentSpec("Text with some InterWordCapitalization",
                    {TokenSpec("Text", 0, 3), TokenSpec("with", 5, 8),
                     TokenSpec("some", 10, 13),
                     TokenSpec("InterWordCapitalization", 15, 37)})});
  Augmentations augmentations = {
      .total = 10, .lowercase = 10, .address = 0, .phone = 0};
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  Augmenter augmenter =
      Augmenter(documents, augmentations, &address_sampler, &phone_sampler);

  augmenter.Augment();

  EXPECT_STREQ(augmenter.documents().documents(0).text().c_str(),
               "Text with some InterWordCapitalization");
  EXPECT_STREQ(augmenter.documents().documents(0).token(0).word().c_str(),
               "Text");
  EXPECT_STREQ(augmenter.documents().documents(0).token(1).word().c_str(),
               "with");
  EXPECT_STREQ(augmenter.documents().documents(0).token(2).word().c_str(),
               "some");
  EXPECT_STREQ(augmenter.documents().documents(0).token(3).word().c_str(),
               "InterWordCapitalization");
  for (int i = 1; i < augmentations.total + 1; ++i) {
    EXPECT_STREQ(augmenter.documents().documents(i).text().c_str(),
                 "text with some interwordcapitalization");
    EXPECT_STREQ(augmenter.documents().documents(i).token(0).word().c_str(),
                 "text");
    EXPECT_STREQ(augmenter.documents().documents(i).token(1).word().c_str(),
                 "with");
    EXPECT_STREQ(augmenter.documents().documents(i).token(2).word().c_str(),
                 "some");
    EXPECT_STREQ(augmenter.documents().documents(i).token(3).word().c_str(),
                 "interwordcapitalization");
  }
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
                     TokenSpec("InterWordCapitalization", 15, 37)}),
       DocumentSpec("Text with some InterWordCapitalization [2]",
                    {TokenSpec("Text", 0, 3), TokenSpec("with", 5, 8),
                     TokenSpec("some", 10, 13),
                     TokenSpec("InterWordCapitalization", 15, 37)})});

  absl::MockingBitGen bitgen;
  // Document sampling.
  EXPECT_CALL(absl::MockUniform<int>(), Call(bitgen, 0, 2))
      .WillOnce(Return(0))
      .WillOnce(Return(1))
      .WillOnce(Return(2))
      .WillOnce(Return(0));
  // Phone/Address replacement probability.
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0))
      .WillRepeatedly(Return(false));
  // Lowercasing.
  InSequence s;
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 2.0 / 4))
      .WillOnce(Return(true));
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 1.0 / 3))
      .WillOnce(Return(false));
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 1.0 / 2))
      .WillOnce(Return(false));
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 1.0 / 1))
      .WillOnce(Return(true));
  Augmentations augmentations = {
      .total = 4, .lowercase = 2, .address = 0, .phone = 0};
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  Augmenter augmenter = Augmenter(documents, augmentations, &address_sampler,
                                  &phone_sampler, bitgen);

  augmenter.Augment();

  ASSERT_EQ(augmenter.documents().documents_size(), 7);
  EXPECT_STREQ(augmenter.documents().documents(3).text().c_str(),
               "text with some interwordcapitalization [0]");
  EXPECT_STREQ(augmenter.documents().documents(4).text().c_str(),
               "Text with some InterWordCapitalization [1]");
  EXPECT_STREQ(augmenter.documents().documents(5).text().c_str(),
               "Text with some InterWordCapitalization [2]");
  EXPECT_STREQ(augmenter.documents().documents(6).text().c_str(),
               "text with some interwordcapitalization [0]");
}

TEST(AugmenterTest, DontLowercaseNonTokens) {
  bert_annotator::Documents documents = ConstructBertDocument(
      {DocumentSpec("[BOS] Text with some InterWordCapitalization [EOS]",
                    {TokenSpec("Text", 6, 9), TokenSpec("with", 11, 14),
                     TokenSpec("some", 16, 19),
                     TokenSpec("InterWordCapitalization", 21, 43)})});
  Augmentations augmentations = {
      .total = 1, .lowercase = 1, .address = 0, .phone = 0};
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  Augmenter augmenter =
      Augmenter(documents, augmentations, &address_sampler, &phone_sampler);

  augmenter.Augment();

  EXPECT_STREQ(augmenter.documents().documents(1).text().c_str(),
               "[BOS] text with some interwordcapitalization [EOS]");
}

TEST(AugmenterTest, DontReplacePhone) {
  bert_annotator::Documents documents = ConstructBertDocument(
      {DocumentSpec("Call 0123456789! Thanks.",
                    {TokenSpec("Call", 0, 3), TokenSpec("0123456789", 5, 14),
                     TokenSpec("Thanks", 17, 22)},
                    {{"lucid", {LabelSpec("TELEPHONE", 1, 1)}}})});
  Augmentations augmentations = {
      .total = 1, .lowercase = 0, .address = 0, .phone = 0};
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  Augmenter augmenter =
      Augmenter(documents, augmentations, &address_sampler, &phone_sampler);

  augmenter.Augment();

  ASSERT_EQ(augmenter.documents().documents_size(), 2);
  ExpectEq(documents.documents(0), augmenter.documents().documents(1));
}

TEST(AugmenterTest, ReplacePhoneSameLength) {
  bert_annotator::Documents documents = ConstructBertDocument(
      {DocumentSpec("Call 0123456789! Thanks.",
                    {TokenSpec("Call", 0, 3), TokenSpec("0123456789", 5, 14),
                     TokenSpec("Thanks", 17, 22)},
                    {{"lucid", {LabelSpec("TELEPHONE", 1, 1)}}})});
  Augmentations augmentations = {
      .total = 1, .lowercase = 0, .address = 0, .phone = 1};
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  std::string replacement = "9876543210";
  EXPECT_CALL(phone_sampler, Sample()).WillOnce(ReturnRef(replacement));
  absl::MockingBitGen bitgen;
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0))
      .Times(2)  // 1 x address, 1 x lowercasing.
      .WillRepeatedly(Return(false));
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 1))
      .WillOnce(Return(true));  // 1 x phone.

  Augmenter augmenter = Augmenter(documents, augmentations, &address_sampler,
                                  &phone_sampler, bitgen);

  augmenter.Augment();

  const bert_annotator::Document augmented = augmenter.documents().documents(1);
  const bert_annotator::Document expected =
      ConstructBertDocument(
          {DocumentSpec(
              "Call 9876543210! Thanks.",
              {TokenSpec("Call", 0, 3), TokenSpec("9876543210", 5, 14),
               TokenSpec("Thanks", 17, 22)},
              {{"lucid", {LabelSpec("TELEPHONE", 1, 1)}}})})
          .documents(0);
  ExpectEq(augmented, expected);
}

TEST(AugmenterTest, ReplacePhoneLongerLength) {
  bert_annotator::Documents documents = ConstructBertDocument(
      {DocumentSpec("Call 0123456789! Thanks.",
                    {TokenSpec("Call", 0, 3), TokenSpec("0123456789", 5, 14),
                     TokenSpec("Thanks", 17, 22)},
                    {{"lucid", {LabelSpec("TELEPHONE", 1, 1)}}})});
  Augmentations augmentations = {
      .total = 1, .lowercase = 0, .address = 0, .phone = 1};
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  std::string replacement = "98765432109876543210";
  EXPECT_CALL(phone_sampler, Sample()).WillOnce(ReturnRef(replacement));
  absl::MockingBitGen bitgen;
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0))
      .Times(2)  // 1 x address, 1 x lowercasing.
      .WillRepeatedly(Return(false));
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 1))
      .WillOnce(Return(true));  // 1 x phone.

  Augmenter augmenter = Augmenter(documents, augmentations, &address_sampler,
                                  &phone_sampler, bitgen);

  augmenter.Augment();

  const bert_annotator::Document augmented = augmenter.documents().documents(1);
  const bert_annotator::Document expected =
      ConstructBertDocument(
          {DocumentSpec("Call 98765432109876543210! Thanks.",
                        {TokenSpec("Call", 0, 3),
                         TokenSpec("98765432109876543210", 5, 24),
                         TokenSpec("Thanks", 27, 32)},
                        {{"lucid", {LabelSpec("TELEPHONE", 1, 1)}}})})
          .documents(0);
  ExpectEq(augmented, expected);
}

TEST(AugmenterTest, ReplacePhoneShorterLength) {
  bert_annotator::Documents documents = ConstructBertDocument(
      {DocumentSpec("Call 0123456789! Thanks.",
                    {TokenSpec("Call", 0, 3), TokenSpec("0123456789", 5, 14),
                     TokenSpec("Thanks", 17, 22)},
                    {{"lucid", {LabelSpec("TELEPHONE", 1, 1)}}})});
  Augmentations augmentations = {
      .total = 1, .lowercase = 0, .address = 0, .phone = 1};
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  std::string replacement = "98";
  EXPECT_CALL(phone_sampler, Sample()).WillOnce(ReturnRef(replacement));
  absl::MockingBitGen bitgen;
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0))
      .Times(2)  // 1 x address, 1 x lowercasing.
      .WillRepeatedly(Return(false));
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 1))
      .WillOnce(Return(true));  // 1 x phone.

  Augmenter augmenter = Augmenter(documents, augmentations, &address_sampler,
                                  &phone_sampler, bitgen);

  augmenter.Augment();

  const bert_annotator::Document augmented = augmenter.documents().documents(1);
  const bert_annotator::Document expected =
      ConstructBertDocument(
          {DocumentSpec("Call 98! Thanks.",
                        {TokenSpec("Call", 0, 3), TokenSpec("98", 5, 6),
                         TokenSpec("Thanks", 9, 14)},
                        {{"lucid", {LabelSpec("TELEPHONE", 1, 1)}}})})
          .documents(0);
  ExpectEq(augmented, expected);
}

TEST(AugmenterTest, ReplacePhoneStart) {
  bert_annotator::Documents documents = ConstructBertDocument({DocumentSpec(
      "0123456789! Thanks.",
      {TokenSpec("0123456789", 0, 9), TokenSpec("Thanks", 12, 17)},
      {{"lucid", {LabelSpec("TELEPHONE", 0, 0)}}})});
  Augmentations augmentations = {
      .total = 1, .lowercase = 0, .address = 0, .phone = 1};
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  std::string replacement = "9876543210";
  EXPECT_CALL(phone_sampler, Sample()).WillOnce(ReturnRef(replacement));
  absl::MockingBitGen bitgen;
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0))
      .Times(2)  // 1 x address, 1 x lowercasing.
      .WillRepeatedly(Return(false));
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 1))
      .WillOnce(Return(true));  // 1 x phone.

  Augmenter augmenter = Augmenter(documents, augmentations, &address_sampler,
                                  &phone_sampler, bitgen);

  augmenter.Augment();

  const bert_annotator::Document augmented = augmenter.documents().documents(1);
  const bert_annotator::Document expected =
      ConstructBertDocument(
          {DocumentSpec(
              "9876543210! Thanks.",
              {TokenSpec("9876543210", 0, 9), TokenSpec("Thanks", 12, 17)},
              {{"lucid", {LabelSpec("TELEPHONE", 0, 0)}}})})
          .documents(0);
  ExpectEq(augmented, expected);
}

TEST(AugmenterTest, ReplacePhoneEnd) {
  bert_annotator::Documents documents = ConstructBertDocument(
      {DocumentSpec("Call 0123456789",
                    {TokenSpec("Call", 0, 3), TokenSpec("0123456789", 5, 14)},
                    {{"lucid", {LabelSpec("TELEPHONE", 1, 1)}}})});
  Augmentations augmentations = {
      .total = 1, .lowercase = 0, .address = 0, .phone = 1};
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  std::string replacement = "9876543210";
  EXPECT_CALL(phone_sampler, Sample()).WillOnce(ReturnRef(replacement));
  absl::MockingBitGen bitgen;
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0))
      .Times(2)  // 1 x address, 1 x lowercasing.
      .WillRepeatedly(Return(false));
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 1))
      .WillOnce(Return(true));  // 1 x phone.

  Augmenter augmenter = Augmenter(documents, augmentations, &address_sampler,
                                  &phone_sampler, bitgen);

  augmenter.Augment();

  const bert_annotator::Document augmented = augmenter.documents().documents(1);
  const bert_annotator::Document expected =
      ConstructBertDocument(
          {DocumentSpec(
              "Call 9876543210",
              {TokenSpec("Call", 0, 3), TokenSpec("9876543210", 5, 14)},
              {{"lucid", {LabelSpec("TELEPHONE", 1, 1)}}})})
          .documents(0);
  ExpectEq(augmented, expected);
}

TEST(AugmenterTest, ReplacePhoneChooseLabel) {
  bert_annotator::Documents documents = ConstructBertDocument({DocumentSpec(
      "0123456789 or 0123456789",
      {TokenSpec("0123456789", 0, 9), TokenSpec("or", 11, 12),
       TokenSpec("0123456789", 14, 23)},
      {{"lucid",
        {LabelSpec("TELEPHONE", 0, 0), LabelSpec("TELEPHONE", 2, 2)}}})});
  Augmentations augmentations = {
      .total = 2, .lowercase = 0, .address = 0, .phone = 2};
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  std::string replacement = "9876543210";
  EXPECT_CALL(phone_sampler, Sample())
      .Times(2)
      .WillRepeatedly(ReturnRef(replacement));
  absl::MockingBitGen bitgen;
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0))
      .Times(4)  // 2 x address, 2 x lowercasing.
      .WillRepeatedly(Return(false));
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 1))
      .Times(2)  // 2 x phone.
      .WillRepeatedly(Return(true));
  // Token selection.
  EXPECT_CALL(absl::MockUniform<size_t>(), Call(bitgen, 0, 1))
      .WillOnce(Return(0))
      .WillOnce(Return(1));

  Augmenter augmenter = Augmenter(documents, augmentations, &address_sampler,
                                  &phone_sampler, bitgen);

  augmenter.Augment();

  bert_annotator::Document augmented = augmenter.documents().documents(1);
  bert_annotator::Document expected =
      ConstructBertDocument(
          {DocumentSpec("9876543210 or 0123456789",
                        {TokenSpec("9876543210", 0, 9), TokenSpec("or", 11, 12),
                         TokenSpec("0123456789", 14, 23)},
                        {{"lucid",
                          {LabelSpec("TELEPHONE", 0, 0),
                           LabelSpec("TELEPHONE", 2, 2)}}})})
          .documents(0);
  ExpectEq(augmented, expected);
  augmented = augmenter.documents().documents(2);
  expected =
      ConstructBertDocument(
          {DocumentSpec("0123456789 or 9876543210",
                        {TokenSpec("0123456789", 0, 9), TokenSpec("or", 11, 12),
                         TokenSpec("9876543210", 14, 23)},
                        {{"lucid",
                          {LabelSpec("TELEPHONE", 0, 0),
                           LabelSpec("TELEPHONE", 2, 2)}}})})
          .documents(0);
  ExpectEq(augmented, expected);
}

TEST(AugmenterTest, ReplacePhoneChooseDocument) {
  bert_annotator::Documents documents = ConstructBertDocument(
      {DocumentSpec("Call 0123456789! Thanks.",
                    {TokenSpec("Call", 0, 3), TokenSpec("0123456789", 5, 14),
                     TokenSpec("Thanks", 17, 22)},
                    {{"lucid", {LabelSpec("TELEPHONE", 1, 1)}}})});
  Augmentations augmentations = {
      .total = 2, .lowercase = 0, .address = 0, .phone = 1};
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  std::string replacement = "9876543210";
  EXPECT_CALL(phone_sampler, Sample()).WillOnce(ReturnRef(replacement));
  absl::MockingBitGen bitgen;
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0))
      .Times(5)  // 2 x address, 2 x lowercasing, 1x phone.
      .WillRepeatedly(Return(false));
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0.5))
      .WillOnce(Return(true));  // Replace phone in first document.
  // Token selection.
  EXPECT_CALL(absl::MockUniform<size_t>(), Call(bitgen, 0, 0))
      .WillOnce(Return(0));

  Augmenter augmenter = Augmenter(documents, augmentations, &address_sampler,
                                  &phone_sampler, bitgen);

  augmenter.Augment();

  bert_annotator::Document augmented = augmenter.documents().documents(1);
  bert_annotator::Document expected =
      ConstructBertDocument(
          {DocumentSpec(
              "Call 9876543210! Thanks.",
              {TokenSpec("Call", 0, 3), TokenSpec("9876543210", 5, 14),
               TokenSpec("Thanks", 17, 22)},
              {{"lucid", {LabelSpec("TELEPHONE", 1, 1)}}})})
          .documents(0);
  ExpectEq(augmented, expected);
  augmented = augmenter.documents().documents(2);
  expected = documents.documents(0);
  ExpectEq(augmented, expected);
}

TEST(AugmenterTest, ReplacePhoneMissingLucid) {
  bert_annotator::Documents documents = ConstructBertDocument(
      {DocumentSpec("MissingLucid", {TokenSpec("MissingLucid", 0, 12)}, {}),
       DocumentSpec("Call 0123456789! Thanks.",
                    {TokenSpec("Call", 0, 3), TokenSpec("0123456789", 5, 14),
                     TokenSpec("Thanks", 17, 22)},
                    {{"lucid", {LabelSpec("TELEPHONE", 1, 1)}}})});
  Augmentations augmentations = {
      .total = 1, .lowercase = 0, .address = 0, .phone = 1};
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  std::string replacement = "9876543210";
  EXPECT_CALL(phone_sampler, Sample()).WillOnce(ReturnRef(replacement));
  absl::MockingBitGen bitgen;
  // Document sampling.
  EXPECT_CALL(absl::MockUniform<int>(), Call(bitgen, 0, 1))
      .WillOnce(Return(0))
      .WillOnce(Return(1));
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0))
      .Times(4)  // 2 x address, 2 x lowercasing.
      .WillRepeatedly(Return(false));
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 1))
      .Times(2)  // 2 x phone.
      .WillRepeatedly(Return(true));

  Augmenter augmenter = Augmenter(documents, augmentations, &address_sampler,
                                  &phone_sampler, bitgen);

  augmenter.Augment();

  ASSERT_EQ(augmenter.documents().documents_size(), 3);
  const bert_annotator::Document augmented = augmenter.documents().documents(2);
  const bert_annotator::Document expected =
      ConstructBertDocument(
          {DocumentSpec(
              "Call 9876543210! Thanks.",
              {TokenSpec("Call", 0, 3), TokenSpec("9876543210", 5, 14),
               TokenSpec("Thanks", 17, 22)},
              {{"lucid", {LabelSpec("TELEPHONE", 1, 1)}}})})
          .documents(0);
  ExpectEq(augmented, expected);
}

// Replacing addresses is very similiar to replacing phone numbers, so not all
// respective tests are repeated here.
TEST(AugmenterTest, DontReplaceAddress) {
  bert_annotator::Documents documents = ConstructBertDocument(
      {DocumentSpec("Visit Zurich! Thanks.",
                    {TokenSpec("Visit", 0, 4), TokenSpec("Zurich", 6, 11),
                     TokenSpec("Thanks", 13, 18)},
                    {{"lucid", {LabelSpec("LOCALITY", 1, 1)}}})});
  Augmentations augmentations = {
      .total = 1, .lowercase = 0, .address = 0, .phone = 0};
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  Augmenter augmenter =
      Augmenter(documents, augmentations, &address_sampler, &phone_sampler);

  augmenter.Augment();

  ASSERT_EQ(augmenter.documents().documents_size(), 2);
  const auto augmented = augmenter.documents().documents(1);
  const auto expected =
      ConstructBertDocument(
          {DocumentSpec("Visit Zurich! Thanks.",
                        {TokenSpec("Visit", 0, 4), TokenSpec("Zurich", 6, 11),
                         TokenSpec("Thanks", 13, 18)},
                        {{"lucid", {LabelSpec("ADDRESS", 1, 1)}}})})
          .documents(0);
  ExpectEq(augmented, expected);
}

TEST(AugmenterTest, UpdateLabels) {
  bert_annotator::Documents documents = ConstructBertDocument(
      {DocumentSpec("Visit Zurich! Thanks.",
                    {TokenSpec("Visit", 0, 4), TokenSpec("Zurich", 6, 11),
                     TokenSpec("Thanks", 13, 18)},
                    {{"lucid", {LabelSpec("LOCALITY", 1, 1)}}})});
  Augmentations augmentations = {
      .total = 0, .lowercase = 0, .address = 0, .phone = 0};
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  Augmenter augmenter =
      Augmenter(documents, augmentations, &address_sampler, &phone_sampler);

  augmenter.Augment();

  ASSERT_EQ(augmenter.documents().documents_size(), 1);
  const auto augmented = augmenter.documents().documents(0);
  const auto expected =
      ConstructBertDocument(
          {DocumentSpec("Visit Zurich! Thanks.",
                        {TokenSpec("Visit", 0, 4), TokenSpec("Zurich", 6, 11),
                         TokenSpec("Thanks", 13, 18)},
                        {{"lucid", {LabelSpec("ADDRESS", 1, 1)}}})})
          .documents(0);
  ExpectEq(augmented, expected);
}

TEST(AugmenterTest, ReplaceAddressSameLength) {
  bert_annotator::Documents documents = ConstructBertDocument(
      {DocumentSpec("Visit Zurich! Thanks.",
                    {TokenSpec("Visit", 0, 4), TokenSpec("Zurich", 6, 11),
                     TokenSpec("Thanks", 13, 18)},
                    {{"lucid", {LabelSpec("LOCALITY", 1, 1)}}})});
  Augmentations augmentations = {
      .total = 1, .lowercase = 0, .address = 1, .phone = 0};
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  std::string replacement = "Munich";
  EXPECT_CALL(address_sampler, Sample()).WillOnce(ReturnRef(replacement));
  absl::MockingBitGen bitgen;
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0))
      .Times(2)  // 1 x address, 1 x lowercasing.
      .WillRepeatedly(Return(false));
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 1))
      .WillOnce(Return(true));  // 1 x phone.

  Augmenter augmenter = Augmenter(documents, augmentations, &address_sampler,
                                  &phone_sampler, bitgen);

  augmenter.Augment();

  const bert_annotator::Document augmented = augmenter.documents().documents(1);
  const bert_annotator::Document expected =
      ConstructBertDocument(
          {DocumentSpec("Visit Munich! Thanks.",
                        {TokenSpec("Visit", 0, 4), TokenSpec("Munich", 6, 11),
                         TokenSpec("Thanks", 13, 18)},
                        {{"lucid", {LabelSpec("ADDRESS", 1, 1)}}})})
          .documents(0);
  ExpectEq(augmented, expected);
}

TEST(AugmenterTest, ReplaceAddressFewerTokens) {
  bert_annotator::Documents documents = ConstructBertDocument({DocumentSpec(
      "Visit Zurich City! Thanks.",
      {TokenSpec("Visit", 0, 4), TokenSpec("Zurich", 6, 11),
       TokenSpec("City", 13, 16), TokenSpec("Thanks", 18, 23)},
      {{"lucid",
        {LabelSpec("LOCALITY", 1, 1), LabelSpec("LOCALITY", 2, 2)}}})});
  Augmentations augmentations = {
      .total = 1, .lowercase = 0, .address = 1, .phone = 0};
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  std::string replacement = "Munich";
  EXPECT_CALL(address_sampler, Sample()).WillOnce(ReturnRef(replacement));
  absl::MockingBitGen bitgen;
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0))
      .Times(2)  // 1 x address, 1 x lowercasing.
      .WillRepeatedly(Return(false));
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 1))
      .WillOnce(Return(true));  // 1 x phone.

  Augmenter augmenter = Augmenter(documents, augmentations, &address_sampler,
                                  &phone_sampler, bitgen);

  augmenter.Augment();

  const bert_annotator::Document augmented = augmenter.documents().documents(1);
  const bert_annotator::Document expected =
      ConstructBertDocument(
          {DocumentSpec("Visit Munich! Thanks.",
                        {TokenSpec("Visit", 0, 4), TokenSpec("Munich", 6, 11),
                         TokenSpec("Thanks", 13, 18)},
                        {{"lucid", {LabelSpec("ADDRESS", 1, 1)}}})})
          .documents(0);
  ExpectEq(augmented, expected);
}

TEST(AugmenterTest, ReplaceAddressMultiWordReplacement) {
  bert_annotator::Documents documents = ConstructBertDocument({DocumentSpec(
      "Visit Zurich City! Thanks.",
      {TokenSpec("Visit", 0, 4), TokenSpec("Zurich", 6, 11),
       TokenSpec("City", 13, 16), TokenSpec("Thanks", 18, 23)},
      {{"lucid",
        {LabelSpec("LOCALITY", 1, 1), LabelSpec("LOCALITY", 2, 2)}}})});
  Augmentations augmentations = {
      .total = 1, .lowercase = 0, .address = 1, .phone = 0};
  MockRandomSampler address_sampler;
  MockRandomSampler phone_sampler;
  std::string replacement = "Munich Centrum";
  EXPECT_CALL(address_sampler, Sample()).WillOnce(ReturnRef(replacement));
  absl::MockingBitGen bitgen;
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 0))
      .Times(2)  // 1 x address, 1 x lowercasing.
      .WillRepeatedly(Return(false));
  EXPECT_CALL(absl::MockBernoulli(), Call(bitgen, 1))
      .WillOnce(Return(true));  // 1 x phone.

  Augmenter augmenter = Augmenter(documents, augmentations, &address_sampler,
                                  &phone_sampler, bitgen);

  augmenter.Augment();

  const bert_annotator::Document augmented = augmenter.documents().documents(1);
  const bert_annotator::Document expected =
      ConstructBertDocument(
          {DocumentSpec(
              "Visit Munich Centrum! Thanks.",
              {TokenSpec("Visit", 0, 4), TokenSpec("Munich Centrum", 6, 19),
               TokenSpec("Thanks", 21, 26)},
              {{"lucid", {LabelSpec("ADDRESS", 1, 1)}}})})
          .documents(0);
  ExpectEq(augmented, expected);
}

}  // namespace augmenter
