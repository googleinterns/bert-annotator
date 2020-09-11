#include "augmenter/augmenter.h"

#include <fcntl.h>

#include <fstream>
#include <iostream>
#include <string>

#include "absl/flags/flag.h"
#include "boost/algorithm/string.hpp"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/text_format.h"
#include "protocol_buffer/document.pb.h"
#include "protocol_buffer/documents.pb.h"

class TextprotoIO {
 public:
  bool load(char* corpus) {
    std::string filename("data/input/preprocessed/");
    filename.append(corpus);
    filename.append(".textproto");

    std::ifstream input(filename);
    google::protobuf::io::IstreamInputStream fileInput(&input,
                                                       std::ios::binary);
    if (!google::protobuf::TextFormat::Parse(&fileInput, &documents_)) {
      std::cerr << "Failed to parse document." << std::endl;
      return false;
    }

    return true;
  }

  bool save(char* corpus) {
    std::string filename("data/output/");
    filename.append(corpus);
    filename.append(".textproto");

    std::ofstream output(filename);
    google::protobuf::io::OstreamOutputStream fileOutput(&output,
                                                         std::ios::binary);
    if (!google::protobuf::TextFormat::Print(documents_, &fileOutput)) {
      std::cerr << "Failed to save document." << std::endl;
      return false;
    }

    return true;
  }

  bert_annotator::Documents get_documents() { return documents_; }

  void set_documents(bert_annotator::Documents documents) {
    documents_ = documents;
  }

 private:
  bert_annotator::Documents documents_;
};

class Augmenter {
 public:
  Augmenter(bert_annotator::Documents documents) { documents_ = documents; }

  // Transforms the text to lowercase
  // Only explicitly listed tokens are transformed
  void set_lowercase(int document_id) {
    bert_annotator::Document* document =
        documents_.mutable_documents(document_id);
    std::string* text = document->mutable_text();
    std::vector<char> new_text_bytes = std::vector<char>();
    int text_index = 0;
    for (int i = 0; i < document->token_size(); i++) {
      bert_annotator::Token* token = document->mutable_token(i);

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
    document->set_text(new_text);
  }

  bert_annotator::Documents get_documents() { return documents_; }

 private:
  bert_annotator::Documents documents_;
};

// Main function:  Reads the entire address book from a file and prints all
//   the information inside.
int main(int argc, char* argv[]) {
  // Verify that the version of the library that we linked against is
  // compatible with the version of the headers we compiled against.
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  if (argc != 2) {
    std::cerr << "Usage:  " << argv[0] << " CORPUS" << std::endl;
    return -1;
  }
  char* corpus = argv[1];

  TextprotoIO textproto_io = TextprotoIO();
  textproto_io.load(corpus);

  Augmenter augmenter = Augmenter(textproto_io.get_documents());
  augmenter.set_lowercase(0);
  augmenter.set_lowercase(1);
  augmenter.set_lowercase(2);
  augmenter.set_lowercase(3);

  textproto_io.set_documents(augmenter.get_documents());
  textproto_io.save(corpus);

  return 0;
}