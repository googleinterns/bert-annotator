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
#include "absl/random/random.h"

// Parses a file containing entities and their corresponding probabilities.
// Expects at least one item in the format [Entitiy]\t[Probability].
// Can be used to draw replacements for the augmentation.
RandomSampler::RandomSampler(std::istringstream& input_stream,
                             absl::BitGenRef bitgenref)
    : bitgenref_(bitgenref) {
  double accumulated_probability = 0;
  random_items_ = std::vector<RandomItem>();

  // Parse input.
  std::string line;
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

    accumulated_probability += probability;
    random_items_.push_back(
        RandomItem(text, probability, accumulated_probability));
  }

  // At least one item needs to exist, otherwise sampling will not be possible.
  if (random_items_.size() == 0) {
    std::cerr << "No item added to the sampler!" << std::endl;
    std::abort();
  }

  // Normalize probabilities.
  for (RandomItem& random_item : random_items_) {
    random_item.Normalize(accumulated_probability);
  }
}

RandomSampler::RandomSampler(std::istringstream& input_stream)
    : RandomSampler(input_stream, bitgen_) {}

const std::string RandomSampler::Sample() {
  double sampled_probability = absl::Uniform<double>(bitgenref_, 0, 1);
  return Search(sampled_probability);
}

std::vector<RandomItem> RandomSampler::items() { return random_items_; }

// Performs a binary search for the first item with target_probability >=
// accumulated_probability.
const std::string RandomSampler::Search(const double target_probability) {
  return Search(target_probability, 0, random_items_.size() - 1);
}

const std::string RandomSampler::Search(const double accumulated_probability,
                                        const int lower_bound,
                                        const int upper_bound) {
  if (lower_bound == upper_bound) {
    return random_items_[lower_bound].text();
  }
  const int center_index = (lower_bound + upper_bound) / 2;
  if (random_items_[center_index].accumulated_probability() <
      accumulated_probability) {
    return Search(accumulated_probability, center_index + 1, upper_bound);
  } else {
    return Search(accumulated_probability, lower_bound, center_index);
  }
}
