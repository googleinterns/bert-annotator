#include "augmenter/augmenter.h"

#include <stdlib.h>

#include <string>

#include "augmenter/percentage.h"
#include "boost/algorithm/string.hpp"
#include "protocol_buffer/document.pb.h"
#include "protocol_buffer/documents.pb.h"

Augmenter::Augmenter(bert_annotator::Documents documents) {
  documents_ = documents;
  srand(time(NULL));
}

// Transforms the text to lowercase
// Only explicitly listed tokens are transformed
void Augmenter::lowercase(Percentage lowercase_percentage) {
  for (bert_annotator::Document& document : *documents_.mutable_documents()) {
    if(lowercase_percentage.percentage > rand() % 100 + 1) {
      continue;
    }

    std::string* text = document.mutable_text();
    std::vector<char> new_text_bytes = std::vector<char>();
    int text_index = 0;
    for (int i = 0; i < document.token_size(); ++i) {
      bert_annotator::Token* token = document.mutable_token(i);

      // Adds the string inbetween two tokens as it is
      int token_start = token->start();
      int token_end = token->end();
      if (text_index < token_start) {
        new_text_bytes.insert(new_text_bytes.end(), text->begin() + text_index,
                              text->begin() + token_start);
      }

      // Transforms the token to lowercase
      std::string* word = token->mutable_word();
      boost::algorithm::to_lower(*word);
      new_text_bytes.insert(new_text_bytes.end(), word->begin(), word->end());
      text_index = token_end + 1;
    }
    new_text_bytes.insert(new_text_bytes.end(), text->begin() + text_index,
                          text->end());
    std::string new_text(new_text_bytes.begin(), new_text_bytes.end());
    document.set_text(new_text);
  }
}

bert_annotator::Documents Augmenter::get_documents() { return documents_; }