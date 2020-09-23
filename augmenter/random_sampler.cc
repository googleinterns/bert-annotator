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

#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

#include "absl/random/bit_gen_ref.h"
#include "absl/random/discrete_distribution.h"
#include "absl/random/random.h"

namespace augmenter {

RandomSampler::RandomSampler(std::istringstream& input_stream) {
  random_items_ = std::vector<RandomItem>();

  // Parse input.
  std::string line;
  std::vector<double> probabilities;
  while (std::getline(input_stream, line)) {
    std::istringstream iss(line);
    std::string text;
    std::getline(iss, text, '\t');
    if (!iss) {
      std::cerr << "Wrong entity format" << std::endl;
      std::abort();
    }
    std::string probability_string;
    std::getline(iss, probability_string, '\n');
    if (!iss) {
      std::cerr << "Wrong entity format" << std::endl;
      std::abort();
    }
    const double probability = std::stod(probability_string);

    random_items_.push_back(RandomItem(text, probability));
    probabilities.push_back(probability);
  }

  // At least one item needs to exist, otherwise sampling will not be possible.
  if (random_items_.size() == 0) {
    std::cerr << "No item added to the sampler!" << std::endl;
    std::abort();
  }

  discrete_distribution_ = absl::discrete_distribution<size_t>(
      probabilities.begin(), probabilities.end());
}

RandomSampler::RandomSampler() {}

const std::string& RandomSampler::Sample() {
  const size_t sampled_index = discrete_distribution_(bitgen_);
  return random_items_[sampled_index].text();
}

const std::vector<RandomItem>& RandomSampler::items() const {
  return random_items_;
}

}  // namespace augmenter
