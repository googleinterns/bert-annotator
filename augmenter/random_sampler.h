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

#ifndef AUGMENTER_RANDOM_SAMPLER_H_
#define AUGMENTER_RANDOM_SAMPLER_H_

#include <string>
#include <utility>
#include <vector>

#include "absl/random/bit_gen_ref.h"
#include "absl/random/discrete_distribution.h"
#include "absl/random/random.h"
#include "augmenter/random_item.h"
#include "gmock/gmock.h"

namespace augmenter {

// Parses a file containing entities and their corresponding probabilities.
// Expects at least one item in the format [Entitiy]\t[Probability].
// Can be used to draw replacements for the augmentation.
class RandomSampler {
 public:
  explicit RandomSampler(std::istream& input_stream);
  virtual const std::string& Sample();
  const std::vector<RandomItem>& items() const;

 protected:
  // Should not be used to construct normal sampler objects, necessary for
  // testing.
  RandomSampler();

 private:
  absl::BitGen bitgen_;
  absl::discrete_distribution<size_t> discrete_distribution_;
  std::vector<RandomItem> random_items_;
};

class MockRandomSampler : public RandomSampler {
 public:
  MockRandomSampler() : RandomSampler() {}
  MOCK_METHOD(const std::string&, Sample, (), (override));
};

}  // namespace augmenter

#endif  // AUGMENTER_RANDOM_SAMPLER_H_
