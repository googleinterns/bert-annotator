#ifndef BERTANNOTATOR_AUGMENTER_TEXTPROTOIO_H_
#define BERTANNOTATOR_AUGMENTER_TEXTPROTOIO_H_
#include <string>

#include "protocol_buffer/documents.pb.h"
class TextprotoIO {
 public:
  bool load(std::string corpus);
  bool save(std::string corpus);
  bert_annotator::Documents get_documents();
  void set_documents(bert_annotator::Documents documents);

 private:
  bert_annotator::Documents documents_;
};
#endif  // BERTANNOTATOR_AUGMENTER_TEXTPROTOIO_H_