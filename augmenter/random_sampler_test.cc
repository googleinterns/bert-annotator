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

#include "augmenter/random_sampler.h"

#include <string>

#include "absl/random/mock_distributions.h"
#include "gtest/gtest.h"

namespace augmenter {

RandomSampler ConstructRandomSampler(const std::string text) {
  std::istringstream dummy_stream(text);
  auto random_sampler = RandomSampler(dummy_stream);
  return random_sampler;
}

RandomSampler ConstructRandomSampler(const std::string text,
                                           absl::BitGenRef bitgenref) {
  std::istringstream dummy_stream(text);
  auto random_sampler = RandomSampler(dummy_stream, bitgenref);
  return random_sampler;
}

TEST(RandomSamplerDeathTest, ErrorOnWrongFormat) {
  EXPECT_DEATH(
      {
        std::istringstream dummyStream("Some text [NoTab] 1");
        auto random_sampler = RandomSampler(dummyStream);
      },
      "Wrong entity format");
}

TEST(RandomSamplerDeathTest, ErrorOnEmptyInput) {
  EXPECT_DEATH(
      {
        std::istringstream dummyStream("");
        auto random_sampler = RandomSampler(dummyStream);
      },
      "No item added to the sampler!");
}

TEST(RandomSamplerTest, ParsingSingleEntry) {
  auto random_sampler = ConstructRandomSampler("Some text\t1");
  std::vector<RandomItem> items = random_sampler.items();
  ASSERT_EQ(items.size(), 1);
  EXPECT_EQ(items[0].text(), "Some text");
  EXPECT_DOUBLE_EQ(items[0].probability(), 1);
  EXPECT_DOUBLE_EQ(items[0].accumulated_probability(), 1);
}

TEST(RandomSamplerTest, ParsingSingleEntryScientific) {
  auto random_sampler = ConstructRandomSampler("Some text\t1e");
  std::vector<RandomItem> items = random_sampler.items();
  ASSERT_EQ(items.size(), 1);
  EXPECT_EQ(items[0].text(), "Some text");
  EXPECT_DOUBLE_EQ(items[0].probability(), 1);
  EXPECT_DOUBLE_EQ(items[0].accumulated_probability(), 1);
}

TEST(RandomSamplerTest, ParsingMultipleEntries) {
  auto random_sampler =
      ConstructRandomSampler("Some text\t0.75\nMore text\t0.25");
  std::vector<RandomItem> items = random_sampler.items();
  ASSERT_EQ(items.size(), 2);
  EXPECT_EQ(items[0].text(), "Some text");
  EXPECT_DOUBLE_EQ(items[0].probability(), 0.75);
  EXPECT_DOUBLE_EQ(items[0].accumulated_probability(), 0.75);
  EXPECT_EQ(items[1].text(), "More text");
  EXPECT_DOUBLE_EQ(items[1].probability(), 0.25);
  EXPECT_DOUBLE_EQ(items[1].accumulated_probability(), 1);
}

TEST(RandomSamplerTest, ParsingMultipleEntriesNormalization) {
  auto random_sampler =
      ConstructRandomSampler("Some text\t0.25\nMore text\t0.25");
  std::vector<RandomItem> items = random_sampler.items();
  ASSERT_EQ(items.size(), 2);
  EXPECT_DOUBLE_EQ(items[0].probability(), 0.5);
  EXPECT_DOUBLE_EQ(items[0].accumulated_probability(), 0.5);
  EXPECT_DOUBLE_EQ(items[1].probability(), 0.5);
  EXPECT_DOUBLE_EQ(items[1].accumulated_probability(), 1);
}

TEST(RandomSamplerTest, SampleSingleEntry) {
  absl::MockingBitGen bitgen;
  EXPECT_CALL(absl::MockUniform<double>(), Call(bitgen, 0.0, 1.0))
      .WillOnce(testing::Return(0.25))
      .WillOnce(testing::Return(0.25))
      .WillOnce(testing::Return(0.25));
  auto random_sampler = ConstructRandomSampler("Some text\t1", bitgen);
  // Sampling multiple times should be possible.
  EXPECT_EQ(random_sampler.Sample(), "Some text");
  EXPECT_EQ(random_sampler.Sample(), "Some text");
  EXPECT_EQ(random_sampler.Sample(), "Some text");
}

TEST(RandomSamplerTest, SampleMultipleEntriesA) {
  absl::MockingBitGen bitgen;
  EXPECT_CALL(absl::MockUniform<double>(), Call(bitgen, 0.0, 1.0))
      .WillOnce(testing::Return(0.25));
  auto random_sampler =
      ConstructRandomSampler("Some text\t0.5\nMore text\t0.5", bitgen);
  EXPECT_EQ(random_sampler.Sample(), "Some text");
}

TEST(RandomSamplerTest, SampleMultipleEntriesB) {
  absl::MockingBitGen bitgen;
  EXPECT_CALL(absl::MockUniform<double>(), Call(bitgen, 0.0, 1.0))
      .WillOnce(testing::Return(0.75));
  auto random_sampler =
      ConstructRandomSampler("Some text\t0.5\nMore text\t0.5", bitgen);
  EXPECT_EQ(random_sampler.Sample(), "More text");
}

}  // namespace augmenter
