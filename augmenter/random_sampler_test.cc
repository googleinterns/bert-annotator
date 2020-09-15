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

#include "gtest/gtest.h"

TEST(RandomSamplerDeathTest, ErrorOnWrongFormat) {
  EXPECT_DEATH(
      {
        std::istringstream dummyStream("Some text [NoTab] 0.5");
        auto random_sampler = RandomSampler(dummyStream);
      },
      "Wrong entity format");
}

TEST(RandomSamplerTest, ParsingSingleEntry) {
  std::istringstream dummyStream("Some text\t0.5");
  auto random_sampler = RandomSampler(dummyStream);
  std::vector<RandomItem> items = random_sampler.items();
  ASSERT_EQ(items.size(), 1);
  EXPECT_EQ(items[0].text(), "Some text");
  EXPECT_EQ(items[0].probability(), 0.5);
  EXPECT_EQ(items[0].accumulated_probability(), 0.5);
}

TEST(RandomSamplerTest, ParsingMultipleEntries) {
  std::istringstream dummyStream("Some text\t0.5\nMore text\t0.25");
  auto random_sampler = RandomSampler(dummyStream);
  std::vector<RandomItem> items = random_sampler.items();
  ASSERT_EQ(items.size(), 2);
  EXPECT_EQ(items[0].text(), "Some text");
  EXPECT_EQ(items[0].probability(), 0.5);
  EXPECT_EQ(items[0].accumulated_probability(), 0.5);
  EXPECT_EQ(items[1].text(), "More text");
  EXPECT_EQ(items[1].probability(), 0.25);
  EXPECT_EQ(items[1].accumulated_probability(), 0.75);
}

