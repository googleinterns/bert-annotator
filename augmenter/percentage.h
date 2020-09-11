#ifndef BERTANNOTATOR_AUGMENTER_PERCENTAGE_H_
#define BERTANNOTATOR_AUGMENTER_PERCENTAGE_H_
#include <string>

#include "absl/flags/flag.h"
struct Percentage {
  explicit Percentage(int p = 0) : percentage(p) {}

  int percentage;  // Valid range is [0..100]
};
std::string AbslUnparseFlag(Percentage p);

bool AbslParseFlag(absl::string_view text, Percentage* p, std::string* error);
#endif  // BERTANNOTATOR_AUGMENTER_PERCENTAGE_H_