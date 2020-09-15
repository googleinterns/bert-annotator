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

#include "augmenter/random_item.h"

RandomItem::RandomItem(std::string text, float probability,
                       float accumulated_probability) {
  text_ = text;
  probability_ = probability;
  accumulated_probability_ = accumulated_probability;
}

std::string RandomItem::text() { return text_; }

float RandomItem::probability() { return probability_; }

float RandomItem::accumulated_probability() { return accumulated_probability_; }