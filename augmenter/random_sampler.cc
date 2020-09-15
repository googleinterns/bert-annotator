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

#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

// Parses a file containing entities and their corresponding probabilities.
// Expects entries in the format [Entitiy]\t[Probability].
// Can be used to draw replacements for the augmentation.
RandomSampler::RandomSampler(std::istringstream& input_stream) {
  float accumulated_probability = 0;
  items_ = std::vector<RandomItem>();

  std::string line;
  while (std::getline(input_stream, line)) {
    std::cout << "LINE READ: " << line << std::endl;
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
    float probability = std::stof(probability_string);

    accumulated_probability += probability;
    items_.push_back(RandomItem(text, probability, accumulated_probability));
  }
}

std::vector<RandomItem> RandomSampler::items() { return items_; }