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
      "No item added to sampler!");
}

TEST(RandomSamplerTest, ParsingSingleEntry) {
  std::istringstream dummyStream("Some text\t1");
  auto random_sampler = RandomSampler(dummyStream);
  std::vector<RandomItem> items = random_sampler.items();
  ASSERT_EQ(items.size(), 1);
  EXPECT_EQ(items[0].text(), "Some text");
  EXPECT_EQ(items[0].probability(), 1);
  EXPECT_EQ(items[0].accumulated_probability(), 1);
}

TEST(RandomSamplerTest, ParsingSingleEntryScientific) {
  std::istringstream dummyStream("Some text\t1e");
  auto random_sampler = RandomSampler(dummyStream);
  std::vector<RandomItem> items = random_sampler.items();
  ASSERT_EQ(items.size(), 1);
  EXPECT_EQ(items[0].text(), "Some text");
  EXPECT_DOUBLE_EQ(items[0].probability(), 1);
  EXPECT_DOUBLE_EQ(items[0].accumulated_probability(), 1);
}

TEST(RandomSamplerTest, ParsingMultipleEntries) {
  std::istringstream dummyStream("Some text\t0.75\nMore text\t0.25");
  auto random_sampler = RandomSampler(dummyStream);
  std::vector<RandomItem> items = random_sampler.items();
  ASSERT_EQ(items.size(), 2);
  EXPECT_EQ(items[0].text(), "Some text");
  EXPECT_EQ(items[0].probability(), 0.75);
  EXPECT_EQ(items[0].accumulated_probability(), 0.75);
  EXPECT_EQ(items[1].text(), "More text");
  EXPECT_EQ(items[1].probability(), 0.25);
  EXPECT_EQ(items[1].accumulated_probability(), 1);
}

TEST(RandomSamplerTest, ParsingMultipleEntriesNormalization) {
  std::istringstream dummyStream("Some text\t0.25\nMore text\t0.25");
  auto random_sampler = RandomSampler(dummyStream);
  std::vector<RandomItem> items = random_sampler.items();
  ASSERT_EQ(items.size(), 2);
  EXPECT_EQ(items[0].probability(), 0.5);
  EXPECT_EQ(items[0].accumulated_probability(), 0.5);
  EXPECT_EQ(items[1].probability(), 0.5);
  EXPECT_EQ(items[1].accumulated_probability(), 1);
}

TEST(RandomSamplerTest, SampleSingleEntry) {
  absl::MockingBitGen bitgen;
  EXPECT_CALL(absl::MockUniform<double>(), Call(bitgen, 0.0, 1.0))
      .WillOnce(testing::Return(0.25))
      .WillOnce(testing::Return(0.25))
      .WillOnce(testing::Return(0.25));
  std::istringstream dummyStream("Some text\t1");
  auto random_sampler = RandomSampler(dummyStream, bitgen);
  // Sampling multiple times should be possible
  EXPECT_EQ(random_sampler.sample(), "Some text");
  EXPECT_EQ(random_sampler.sample(), "Some text");
  EXPECT_EQ(random_sampler.sample(), "Some text");
}

TEST(RandomSamplerTest, SampleMultipleEntriesA) {
  absl::MockingBitGen bitgen;
  EXPECT_CALL(absl::MockUniform<double>(), Call(bitgen, 0.0, 1.0))
      .WillOnce(testing::Return(0.25));
  std::istringstream dummyStream("Some text\t0.5\nMore text\t0.5");
  auto random_sampler = RandomSampler(dummyStream, bitgen);
  EXPECT_EQ(random_sampler.sample(), "Some text");
}

TEST(RandomSamplerTest, SampleMultipleEntriesB) {
  absl::MockingBitGen bitgen;
  EXPECT_CALL(absl::MockUniform<double>(), Call(bitgen, 0.0, 1.0))
      .WillOnce(testing::Return(0.75));
  std::istringstream dummyStream("Some text\t0.5\nMore text\t0.5");
  auto random_sampler = RandomSampler(dummyStream, bitgen);
  EXPECT_EQ(random_sampler.sample(), "More text");
}
