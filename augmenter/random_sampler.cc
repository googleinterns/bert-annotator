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
RandomSampler::RandomSampler(std::istringstream& input_stream) {
  double accumulated_probability = 0;
  items_ = std::vector<RandomItem>();

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
    double probability = std::stod(probability_string);

    accumulated_probability += probability;
    items_.push_back(RandomItem(text, probability, accumulated_probability));
  }

  // At least one item needs to exist, otherwise sampling will not be possible
  if (items_.size() == 0) {
    std::cerr << "No item added to sampler!" << std::endl;
    std::abort();
  }

  // Normalize probabilities.
  for (auto& random_item : items_) {
    random_item.normalize(accumulated_probability);
  }
}

std::string RandomSampler::sample() {
  double sampled_probability = absl::Uniform(bitgen_, 0, 1.0);
  return search(sampled_probability);
}
std::string RandomSampler::sample(absl::BitGen bitgen) {
  // bitgen_ = bitgen; NOT POSSIBLE
  double sampled_probability = absl::Uniform(bitgen, 0, 1.0);
  return search(sampled_probability);
}

std::vector<RandomItem> RandomSampler::items() { return items_; }

std::string RandomSampler::search(double accumulated_probability) {
  return search(accumulated_probability, 0, items_.size() - 1);
}
std::string RandomSampler::search(double accumulated_probability,
                                  int lower_bound, int upper_bound) {
  if (lower_bound == upper_bound) {
    return items_[lower_bound].text();
  }
  int center_index = (lower_bound + upper_bound) / 2;
  if (items_[center_index].accumulated_probability() <
      accumulated_probability) {
    return search(accumulated_probability, center_index + 1, upper_bound);
  } else {
    return search(accumulated_probability, lower_bound, center_index);
  }
}
